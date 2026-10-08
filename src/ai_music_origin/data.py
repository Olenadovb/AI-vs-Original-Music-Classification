"""Loading and preprocessing audio data for the AI music origin project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load an audio file from disk.
    Parameters
    path : str or Path
        Path to the audio file.

    Returns
    tuple[np.ndarray, int]
        Audio signal and sample rate.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sample_rate = sf.read(path, dtype="float32")
    return audio, sample_rate


def trim_or_pad(audio: np.ndarray, sample_rate: int, target_duration: float,) -> np.ndarray:
    """
    Trim or pad an audio signal to a fixed duration.
    Parameters
    audio : np.ndarray
        Input audio signal.
    sample_rate : int
        Sample rate in Hz.
    target_duration : float
        Target duration in seconds.

    Returns
    np.ndarray
        Audio signal with the requested duration.
    """
    target_length = int(sample_rate * target_duration)

    if len(audio) > target_length:
        return audio[:target_length]

    if len(audio) < target_length:
        padding = target_length - len(audio)
        return np.pad(audio, (0, padding))

    return audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Scale audio so its maximum absolute amplitude is 1.
    Parameters
    audio : np.ndarray
        Input audio signal.

    Returns
    np.ndarray
        Normalized audio signal.
    """
    max_amplitude = np.max(np.abs(audio))

    if max_amplitude == 0:
        return audio

    return audio / max_amplitude


def preprocess_audio(audio: np.ndarray, sample_rate: int, target_duration: float | None = None, normalize: bool = False,) -> np.ndarray:
    """
    Apply optional preprocessing steps to an audio signal.
    Parameters
    audio : np.ndarray
        Input audio signal.
    sample_rate : int
        Sample rate in Hz.
    target_duration : float or None
        If provided, trim or pad audio to this duration.
    normalize : bool
        Whether to normalize audio amplitude.

    Returns
    np.ndarray
        Processed audio signal.
    """
    processed_audio = audio.copy()

    if target_duration is not None:
        processed_audio = trim_or_pad(processed_audio, sample_rate, target_duration)

    if normalize:
        processed_audio = normalize_audio(processed_audio)

    return processed_audio
