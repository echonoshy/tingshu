from argparse import Namespace
import inspect
from io import BytesIO
import unittest
from unittest.mock import patch

import numpy as np
import soundfile


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_gradio_interface_builds_with_the_locked_version(self):
        from tools import webui

        with patch.object(webui, "args", Namespace(theme="light"), create=True):
            app = webui.build_app()
        inspect.signature(app.launch).bind(theme=webui.gr.themes.Base())

    def test_reference_audio_decodes_to_resampled_mono(self):
        from tools.audio import load_mono_audio

        input_rate = 8000
        output_rate = 16000
        samples = 0.2 * np.sin(np.arange(input_rate) / 12)
        audio = BytesIO()
        soundfile.write(audio, np.column_stack((samples, samples / 2)), input_rate, format="WAV")

        waveform = load_mono_audio(audio.getvalue(), output_rate)
        self.assertEqual((waveform.shape, waveform.dtype), ((output_rate,), np.float32))
        np.testing.assert_allclose(waveform[2:-2:2], 0.75 * samples[1:-1], atol=0.002)
