from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _logit(value: float) -> float:
    value = max(1e-4, min(1 - 1e-4, value))
    return math.log(value / (1.0 - value))


class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        groups_for_norm = min(8, out_channels)
        while out_channels % groups_for_norm != 0:
            groups_for_norm -= 1
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(groups_for_norm, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBnAct(in_channels, in_channels, 3, stride=stride, dilation=dilation, groups=in_channels),
            ConvBnAct(in_channels, out_channels, 1),
        )
        self.skip = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.skip:
            out = out + x
        return out


class GlobalContext(nn.Module):
    def __init__(self, channels: int, squeeze: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // squeeze)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.gate(self.pool(x))
        return x * (1.0 + scale)


def detector_channel_count(
    num_classes: int = 1,
    use_centerness: bool = False,
    class_heatmap: bool = False,
    class_box: bool = False,
) -> int:
    if class_heatmap and int(num_classes) > 1:
        box_channels = int(num_classes) * 4 if class_box else 4
        return int(num_classes) + box_channels + (1 if use_centerness else 0)
    class_channels = max(0, int(num_classes)) if int(num_classes) > 1 else 0
    return 5 + (1 if use_centerness else 0) + class_channels


def _head_bias(
    out_channels: int,
    *,
    num_classes: int = 1,
    use_centerness: bool = False,
    class_heatmap: bool = False,
    class_box: bool = False,
    size_prior: tuple[float, float] = (0.09, 0.12),
) -> torch.Tensor:
    bias = torch.zeros(out_channels, dtype=torch.float32)
    if class_heatmap and int(num_classes) > 1:
        heatmap_channels = int(num_classes)
        bias[:heatmap_channels] = -4.6
        box_start = heatmap_channels
        box_channels = heatmap_channels * 4 if class_box else 4
        if class_box:
            for class_index in range(heatmap_channels):
                class_box_start = box_start + class_index * 4
                bias[class_box_start + 2] = _logit(size_prior[0])
                bias[class_box_start + 3] = _logit(size_prior[1])
        else:
            bias[box_start + 2] = _logit(size_prior[0])
            bias[box_start + 3] = _logit(size_prior[1])
        if use_centerness and out_channels > box_start + box_channels:
            bias[box_start + box_channels] = -2.2
    else:
        bias[0] = -4.6
        bias[3] = _logit(size_prior[0])
        bias[4] = _logit(size_prior[1])
        if use_centerness and out_channels > 5:
            bias[5] = -2.2
        if num_classes > 1:
            class_start = 6 if use_centerness else 5
            if class_start < out_channels:
                bias[class_start:] = 0.0
    return bias


def _init_detector_head(
    head: nn.Module,
    *,
    out_channels: int,
    num_classes: int = 1,
    use_centerness: bool = False,
    class_heatmap: bool = False,
    class_box: bool = False,
    size_prior: tuple[float, float] = (0.09, 0.12),
) -> None:
    final = head[-1] if isinstance(head, nn.Sequential) else head
    if isinstance(final, nn.Conv2d):
        nn.init.normal_(final.weight, std=0.01)
        with torch.no_grad():
            final.bias.copy_(
                _head_bias(
                    out_channels,
                    num_classes=num_classes,
                    use_centerness=use_centerness,
                    class_heatmap=class_heatmap,
                    class_box=class_box,
                    size_prior=size_prior,
                )
            )


def _softplus_inverse(value: float) -> float:
    value = max(1e-4, float(value))
    return math.log(math.expm1(value))


def _init_quality_head(head: nn.Module, num_classes: int, distance_prior: float) -> None:
    for module in head.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    if isinstance(head.cls_logits, nn.Conv2d):
        with torch.no_grad():
            head.cls_logits.bias.fill_(-4.6)
    if isinstance(head.box_reg, nn.Conv2d):
        with torch.no_grad():
            head.box_reg.bias.fill_(_softplus_inverse(distance_prior))
    if isinstance(head.quality_logits, nn.Conv2d):
        with torch.no_grad():
            head.quality_logits.bias.fill_(-2.2)


class QualityDetectionHead(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        distance_prior: float = 4.0,
        center_offset: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.center_offset = bool(center_offset)
        self.cls_tower = nn.Sequential(
            DepthwiseSeparable(channels, channels),
            ConvBnAct(channels, channels, 1),
        )
        self.reg_tower = nn.Sequential(
            DepthwiseSeparable(channels, channels),
            ConvBnAct(channels, channels, 1),
        )
        self.cls_logits = nn.Conv2d(channels, self.num_classes, 1)
        self.offset_reg = nn.Conv2d(channels, 2, 1) if self.center_offset else None
        self.box_reg = nn.Conv2d(channels, 4, 1)
        self.quality_logits = nn.Conv2d(channels, 1, 1)
        _init_quality_head(self, self.num_classes, distance_prior)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        cls_feature = self.cls_tower(feature)
        reg_feature = self.reg_tower(feature)
        outputs = [self.cls_logits(cls_feature)]
        if self.offset_reg is not None:
            outputs.append(self.offset_reg(reg_feature))
        outputs.extend([self.box_reg(reg_feature), self.quality_logits(reg_feature)])
        return torch.cat(
            outputs,
            dim=1,
        )


class TinyContextDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        use_centerness: bool = False,
        class_heatmap: bool = False,
        class_box: bool = False,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_centerness = bool(use_centerness)
        self.class_heatmap = bool(class_heatmap)
        self.class_box = bool(class_box)
        self.out_channels = int(
            out_channels
            or detector_channel_count(self.num_classes, self.use_centerness, self.class_heatmap, self.class_box)
        )
        self.backbone = nn.Sequential(
            ConvBnAct(3, 16, 3, stride=2),
            DepthwiseSeparable(16, 24, stride=2),
            DepthwiseSeparable(24, 32),
            DepthwiseSeparable(32, 48, stride=2),
            DepthwiseSeparable(48, 64),
            DepthwiseSeparable(64, 96, dilation=2),
            DepthwiseSeparable(96, 96, dilation=4),
            DepthwiseSeparable(96, 96, dilation=6),
            GlobalContext(96),
        )
        self.head = nn.Sequential(
            ConvBnAct(96, 80, 1),
            nn.Conv2d(80, self.out_channels, 1),
        )
        self._init_head()

    def _init_head(self) -> None:
        _init_detector_head(
            self.head,
            out_channels=self.out_channels,
            num_classes=self.num_classes,
            use_centerness=self.use_centerness,
            class_heatmap=self.class_heatmap,
            class_box=self.class_box,
            size_prior=(0.09, 0.12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class MobileNetContextDetector(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 1,
        use_centerness: bool = False,
        class_heatmap: bool = False,
        class_box: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_centerness = bool(use_centerness)
        self.class_heatmap = bool(class_heatmap)
        self.class_box = bool(class_box)
        self.out_channels = detector_channel_count(
            self.num_classes,
            self.use_centerness,
            self.class_heatmap,
            self.class_box,
        )
        try:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except ImportError as exc:
            raise RuntimeError("mobilenet_context requires torchvision to be installed") from exc

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        features = list(mobilenet_v3_small(weights=weights).features.children())
        self.stage8 = nn.Sequential(*features[:4])
        self.stage16 = nn.Sequential(*features[4:9])
        self.context = nn.Sequential(
            ConvBnAct(72, 96, 1),
            DepthwiseSeparable(96, 96, dilation=2),
            DepthwiseSeparable(96, 96, dilation=4),
            GlobalContext(96),
        )
        self.head = nn.Sequential(
            ConvBnAct(96, 80, 1),
            nn.Conv2d(80, self.out_channels, 1),
        )
        self._init_head()

    def _init_head(self) -> None:
        _init_detector_head(
            self.head,
            out_channels=self.out_channels,
            num_classes=self.num_classes,
            use_centerness=self.use_centerness,
            class_heatmap=self.class_heatmap,
            class_box=self.class_box,
            size_prior=(0.09, 0.12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride8 = self.stage8(x)
        stride16 = self.stage16(stride8)
        stride16 = F.interpolate(stride16, size=stride8.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.context(torch.cat([stride8, stride16], dim=1)))


class MultiScaleContextDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        use_centerness: bool = True,
        class_heatmap: bool = False,
        class_box: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_centerness = bool(use_centerness)
        self.class_heatmap = bool(class_heatmap)
        self.class_box = bool(class_box)
        self.out_channels = detector_channel_count(
            self.num_classes,
            self.use_centerness,
            self.class_heatmap,
            self.class_box,
        )
        self.stem = ConvBnAct(3, 16, 3, stride=2)
        self.stage4 = nn.Sequential(
            DepthwiseSeparable(16, 24, stride=2),
            DepthwiseSeparable(24, 32),
        )
        self.stage8 = nn.Sequential(
            DepthwiseSeparable(32, 48, stride=2),
            DepthwiseSeparable(48, 64),
        )
        self.stage16 = nn.Sequential(
            DepthwiseSeparable(64, 96, stride=2),
            DepthwiseSeparable(96, 96, dilation=2),
            DepthwiseSeparable(96, 96, dilation=4),
            GlobalContext(96),
        )
        self.fuse8 = nn.Sequential(
            ConvBnAct(160, 96, 1),
            DepthwiseSeparable(96, 96, dilation=2),
            GlobalContext(96),
        )
        self.fuse4 = nn.Sequential(
            ConvBnAct(128, 64, 1),
            DepthwiseSeparable(64, 64),
            GlobalContext(64),
        )
        self.head4 = nn.Sequential(
            ConvBnAct(64, 64, 1),
            nn.Conv2d(64, self.out_channels, 1),
        )
        self.head8 = nn.Sequential(
            ConvBnAct(96, 80, 1),
            nn.Conv2d(80, self.out_channels, 1),
        )
        self._init_heads()

    def _init_heads(self) -> None:
        _init_detector_head(
            self.head4,
            out_channels=self.out_channels,
            num_classes=self.num_classes,
            use_centerness=self.use_centerness,
            class_heatmap=self.class_heatmap,
            class_box=self.class_box,
            size_prior=(0.028, 0.028),
        )
        _init_detector_head(
            self.head8,
            out_channels=self.out_channels,
            num_classes=self.num_classes,
            use_centerness=self.use_centerness,
            class_heatmap=self.class_heatmap,
            class_box=self.class_box,
            size_prior=(0.07, 0.07),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(x)
        stride4 = self.stage4(stem)
        stride8 = self.stage8(stride4)
        stride16 = self.stage16(stride8)
        context8 = F.interpolate(stride16, size=stride8.shape[-2:], mode="bilinear", align_corners=False)
        fused8 = self.fuse8(torch.cat([stride8, context8], dim=1))
        context4 = F.interpolate(fused8, size=stride4.shape[-2:], mode="bilinear", align_corners=False)
        fused4 = self.fuse4(torch.cat([stride4, context4], dim=1))
        return {"p4": self.head4(fused4), "p8": self.head8(fused8)}


class FPNQualityDetector(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 1,
        fpn_channels: int = 96,
        center_offset: bool = False,
        panet: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_centerness = True
        self.class_heatmap = True
        self.class_box = False
        self.quality_fpn = True
        self.center_offset = bool(center_offset)
        self.nwd_loss = bool(center_offset)
        self.task_aligned = False
        self.panet = bool(panet)
        self.advanced_box_loss = False
        self.box_format = "center_ltrb" if self.center_offset else "ltrb"
        self.out_channels = self.num_classes + 5 + (2 if self.center_offset else 0)
        try:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except ImportError as exc:
            raise RuntimeError("fpn_quality requires torchvision to be installed") from exc

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        features = list(mobilenet_v3_small(weights=weights).features.children())
        self.stage4 = nn.Sequential(*features[:2])
        self.stage8 = nn.Sequential(*features[2:4])
        self.stage16 = nn.Sequential(*features[4:9])
        self.stage32 = nn.Sequential(*features[9:])

        self.lat4 = ConvBnAct(16, fpn_channels, 1)
        self.lat8 = ConvBnAct(24, fpn_channels, 1)
        self.lat16 = ConvBnAct(48, fpn_channels, 1)
        self.lat32 = ConvBnAct(576, fpn_channels, 1)
        self.out4 = nn.Sequential(DepthwiseSeparable(fpn_channels, fpn_channels), GlobalContext(fpn_channels))
        self.out8 = nn.Sequential(DepthwiseSeparable(fpn_channels, fpn_channels), GlobalContext(fpn_channels))
        self.out16 = nn.Sequential(DepthwiseSeparable(fpn_channels, fpn_channels), GlobalContext(fpn_channels))
        self.out32 = nn.Sequential(DepthwiseSeparable(fpn_channels, fpn_channels), GlobalContext(fpn_channels))
        if self.panet:
            self.down4 = DepthwiseSeparable(fpn_channels, fpn_channels, stride=2)
            self.down8 = DepthwiseSeparable(fpn_channels, fpn_channels, stride=2)
            self.down16 = DepthwiseSeparable(fpn_channels, fpn_channels, stride=2)
            self.pan8 = nn.Sequential(DepthwiseSeparable(fpn_channels, fpn_channels), GlobalContext(fpn_channels))
            self.pan16 = nn.Sequential(DepthwiseSeparable(fpn_channels, fpn_channels), GlobalContext(fpn_channels))
            self.pan32 = nn.Sequential(DepthwiseSeparable(fpn_channels, fpn_channels), GlobalContext(fpn_channels))

        self.head4 = QualityDetectionHead(
            fpn_channels, self.num_classes, distance_prior=2.0, center_offset=self.center_offset
        )
        self.head8 = QualityDetectionHead(
            fpn_channels, self.num_classes, distance_prior=3.5, center_offset=self.center_offset
        )
        self.head16 = QualityDetectionHead(
            fpn_channels, self.num_classes, distance_prior=5.0, center_offset=self.center_offset
        )
        self.head32 = QualityDetectionHead(
            fpn_channels, self.num_classes, distance_prior=7.0, center_offset=self.center_offset
        )

    @staticmethod
    def _upsample_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c4 = self.stage4(x)
        c8 = self.stage8(c4)
        c16 = self.stage16(c8)
        c32 = self.stage32(c16)

        p32 = self.lat32(c32)
        p16 = self.lat16(c16) + self._upsample_like(p32, c16)
        p8 = self.lat8(c8) + self._upsample_like(p16, c8)
        p4 = self.lat4(c4) + self._upsample_like(p8, c4)

        if self.panet:
            p4 = self.out4(p4)
            p8 = self.pan8(self.out8(p8) + self._upsample_like(self.down4(p4), p8))
            p16 = self.pan16(self.out16(p16) + self._upsample_like(self.down8(p8), p16))
            p32 = self.pan32(self.out32(p32) + self._upsample_like(self.down16(p16), p32))
            return {
                "p4": self.head4(p4),
                "p8": self.head8(p8),
                "p16": self.head16(p16),
                "p32": self.head32(p32),
            }

        return {
            "p4": self.head4(self.out4(p4)),
            "p8": self.head8(self.out8(p8)),
            "p16": self.head16(self.out16(p16)),
            "p32": self.head32(self.out32(p32)),
        }


def create_detector_model(
    model_name: str = "tiny_context",
    pretrained: bool = False,
    num_classes: int = 1,
    use_centerness: bool = False,
    class_heatmap: bool = False,
    class_box: bool = False,
) -> nn.Module:
    if model_name == "tiny_context":
        return TinyContextDetector(
            num_classes=num_classes,
            use_centerness=use_centerness,
            class_heatmap=class_heatmap,
            class_box=class_box,
        )
    if model_name == "mobilenet_context":
        return MobileNetContextDetector(
            pretrained=pretrained,
            num_classes=num_classes,
            use_centerness=use_centerness,
            class_heatmap=class_heatmap,
            class_box=class_box,
        )
    if model_name == "multiscale_context":
        return MultiScaleContextDetector(
            num_classes=num_classes,
            use_centerness=True if use_centerness is False else use_centerness,
            class_heatmap=class_heatmap,
            class_box=class_box,
        )
    if model_name in {"fpn_quality", "fpn_quality_v2", "fpn_quality_v3", "fpn_quality_v4"}:
        model = FPNQualityDetector(
            pretrained=pretrained,
            num_classes=num_classes,
            center_offset=model_name in {"fpn_quality_v2", "fpn_quality_v3", "fpn_quality_v4"},
            panet=model_name == "fpn_quality_v4",
        )
        model.task_aligned = model_name in {"fpn_quality_v3", "fpn_quality_v4"}
        model.advanced_box_loss = model_name == "fpn_quality_v4"
        return model
    raise ValueError(
        "neural model must be 'tiny_context', 'mobilenet_context', 'multiscale_context', 'fpn_quality', "
        "'fpn_quality_v2', 'fpn_quality_v3', or 'fpn_quality_v4'"
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "ConvBnAct",
    "DepthwiseSeparable",
    "FPNQualityDetector",
    "GlobalContext",
    "MobileNetContextDetector",
    "MultiScaleContextDetector",
    "TinyContextDetector",
    "count_parameters",
    "create_detector_model",
    "detector_channel_count",
]
