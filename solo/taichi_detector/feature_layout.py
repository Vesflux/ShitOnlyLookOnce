from __future__ import annotations


FEATURE_SIZE = 14
FEATURE_CHANNELS = 5
QUALITY_FEATURES = 8
STAT_FEATURES = 8


def feature_dimension(feature_size: int = FEATURE_SIZE) -> int:
    visual_features = 2 * FEATURE_CHANNELS * feature_size * feature_size
    statistical_features = STAT_FEATURES * 3
    geometry_features = 10
    return visual_features + statistical_features + QUALITY_FEATURES + geometry_features


__all__ = [
    "FEATURE_CHANNELS",
    "FEATURE_SIZE",
    "QUALITY_FEATURES",
    "STAT_FEATURES",
    "feature_dimension",
]
