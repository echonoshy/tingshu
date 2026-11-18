"""Decode reference audio and prepare a mono float32 waveform."""

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from torchaudio.functional import resample
import torch


def load_mono_audio(source: str | Path | bytes, sample_rate: int) -> np.ndarray:
    media = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    try:
        samples, original_rate = sf.read(media, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.mean(axis=1))
        if original_rate != sample_rate:
            waveform = resample(waveform, original_rate, sample_rate)
        return waveform.numpy()
    except sf.LibsndfileError:
        if isinstance(media, io.BytesIO):
            media.seek(0)
        # FFmpeg covers formats such as AAC and M4A that libsndfile may not read.
        clip = AudioSegment.from_file(media).set_channels(1).set_frame_rate(sample_rate)
        return np.asarray(clip.get_array_of_samples(), dtype=np.float32) / clip.max_possible_amplitude
