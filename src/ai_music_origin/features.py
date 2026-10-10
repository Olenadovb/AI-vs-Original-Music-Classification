"""Handcrafted acoustic feature extraction."""
from __future__ import annotations
import librosa
import numpy as np

import pandas as pd

def extract_handcrafted_features(audio: np.ndarray, sample_rate: int,) -> dict[str, float]:
    features = {}

    rms = librosa.feature.rms(y=audio)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    zero = librosa.feature.zero_crossing_rate(audio)[0]
    features["zero_mean"] = float(np.mean(zero))
    features["zero_std"] = float(np.std(zero))

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

    for i in range(mfcc.shape[0]):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    return features
