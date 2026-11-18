import gc
import queue

import numpy as np
import pydub
import torch
from loguru import logger

from src.text.chn_text_norm.text import Text as ChnNormedText
from src.utils import autocast_exclude_mps, set_seed
from tools.api import decode_vq_tokens, encode_reference
from tools.llama.generate import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
    launch_thread_safe_queue,
)
from tools.vqgan.inference import load_model as load_decoder_model


class SpeechInference:
    def __init__(
        self,
        llama_checkpoint_path: str = "weights",
        decoder_checkpoint_path: str = "weights/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        decoder_config_name: str = "firefly_gan_vq",
        device: str | None = None,
        precision: torch.dtype | None = None,
        compile: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if precision is None:
            precision = torch.float32 if device == "cpu" else torch.bfloat16
        self.device = device
        self.precision = precision
        self.compile = compile

        # Initialize models
        logger.info("Loading Llama model...")
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=llama_checkpoint_path,
            device=device,
            precision=precision,
            compile=compile,
        )

        logger.info("Llama model loaded, loading VQ-GAN model...")
        self.decoder_model = load_decoder_model(
            config_name=decoder_config_name,
            checkpoint_path=decoder_checkpoint_path,
            device=device,
        )

        logger.info("Models loaded successfully")

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        enable_reference_audio: bool = True,
        reference_audio = None,
        reference_text: str = "",
        max_new_tokens: int = 0,
        chunk_length: int = 200,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        temperature: float = 0.7,
        seed: int = 0,
    ):
        if seed != 0:
            set_seed(seed)
            logger.warning(f"Set seed: {seed}")

        # Parse reference audio/prompt
        prompt_tokens = encode_reference(
            decoder_model=self.decoder_model,
            reference_audio=reference_audio,
            enable_reference_audio=enable_reference_audio,
        )

        # Prepare LLAMA inference request
        request = dict(
            device=self.decoder_model.device,
            max_new_tokens=max_new_tokens,
            text=self.normalize_text(text),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=self.compile,
            iterative_prompt=chunk_length > 0,
            chunk_length=chunk_length,
            max_length=2048,
            prompt_tokens=prompt_tokens if enable_reference_audio else None,
            prompt_text=reference_text if enable_reference_audio else None,
        )

        response_queue = queue.Queue()
        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        segments = []

        while True:
            result: WrappedGenerateResponse = response_queue.get()
            if result.status == "error":
                raise RuntimeError(f"Generation failed: {result.response}")
                # logger.info("skip problem segment. ")
                # continue

            result: GenerateResponse = result.response
            if result.action == "next":
                break

            with autocast_exclude_mps(
                device_type=self.decoder_model.device.type,
                dtype=self.precision
            ):
                fake_audios = decode_vq_tokens(
                    decoder_model=self.decoder_model,
                    codes=result.codes,
                )

            fake_audios = fake_audios.float().cpu().numpy()
            segments.append(fake_audios)

        if len(segments) == 0:
            raise RuntimeError("No audio generated, please check the input text.")

        # Concatenate all audio segments
        audio = np.concatenate(segments, axis=0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return self.decoder_model.spec_transform.sample_rate, audio

    def normalize_text(self, text: str, use_normalization: bool = False):
        """Normalize input text (currently only supports Chinese)"""
        if use_normalization:
            return ChnNormedText(raw_text=text).normalize()
        return text


def bytes_to_audio(audio:bytes, sample_rate:int, output_file, format:str="mp3"):
    audio = (audio * 32767).astype(np.int16)
    audio_segment = pydub.AudioSegment(
        audio.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio.dtype.itemsize,
        channels=1                              # 单声道
    )
    audio_segment = audio_segment.set_frame_rate(16000)         # 降采样，减少文件大小
    audio_segment.export(output_file, format=format)
