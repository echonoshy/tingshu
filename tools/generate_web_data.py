
import click 
import json 
import gc
import os
import sys 
import numpy as np
import torch
import queue
import pydub 
from loguru import logger
from pathlib import Path

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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
        device: str = "cuda",
        precision = torch.bfloat16,
        compile: bool = True
    ):
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


def load_text_files(text_dir: Path):
    """Loads all text files and returns a list of file names and content."""
    text_files = []
    for text_path in text_dir.glob("*.txt"):
        with open(text_path, "r") as file:
            text_content = file.read()
        text_files.append((text_path.stem, text_content))
    return text_files

def generate_audio_from_text(model, text_content: str, reference_audio: Path, reference_text: str):
    """Generates audio using the model and returns the sample rate and audio data."""
    sample_rate, audio = model.generate(
        text=text_content,
        reference_audio=str(reference_audio),
        reference_text=reference_text
    )
    return sample_rate, audio

def save_audio(audio, sample_rate: int, output_path: Path, format="mp3"):
    """Saves audio data to a specified file format."""
    os.makedirs(output_path.parent, exist_ok=True)
    bytes_to_audio(audio, sample_rate, output_file=str(output_path), format=format)

def generate_info_json(resources_dir: Path, output_json: Path):
    """Generates a JSON file with audio metadata based on the generated mp3 files."""
    text_dir = resources_dir / "books"
    audio_dir_orig = resources_dir / "audios"      # TIP: web的根路径和项目根路径不一致
    audio_dir = Path(*audio_dir_orig.parts[1:])
    
    # Initialize dictionary structure for JSON
    info_data = {"audios": []}

    # Loop through each text file to create audio info entries
    for text_path in sorted(text_dir.glob("*.txt")):
        text_name = text_path.stem  # Get filename without extension
        audio_src = audio_dir / f"{text_name}.mp3"  # Define audio path
        
        # Add audio entry to list
        info_data["audios"].append({
            "title": text_name,
            "src": str(audio_src)
        })

    # Write the dictionary to a JSON file
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(info_data, json_file, ensure_ascii=False, indent=4)
    
    logger.info(f"info.json has been generated at {output_json}")


@click.command()
@click.option("--resources_dir", required=True, help="Path to the resources directory")
def main(resources_dir: str):
    model = SpeechInference()

    # Define resource directories and files
    resources_dir = Path(resources_dir)
    text_dir = resources_dir / "books"
    reference_audio = resources_dir / "reference.wav"
    reference_text_path = resources_dir / "reference.txt"
    output_dir = resources_dir / "audios"
    output_json = resources_dir / "info.json"

    # Check if resource files exist
    if not reference_audio.exists() or not reference_text_path.exists():
        logger.error("Error: Reference audio or text file is missing.")
        return

    # Read the reference text content
    try:
        with open(reference_text_path, "r") as f_ref:
            reference_text = f_ref.read()
    except IOError as e:
        logger.error(f"Error reading reference text file: {e}")
        return

    # Iterate over and generate audio for each text file
    text_files = load_text_files(text_dir)
    for text_name, text_content in text_files:
        try:
            sample_rate, audio = generate_audio_from_text(
                model, text_content, reference_audio, reference_text
            )
            output_file = output_dir / f"{text_name}.mp3"
            save_audio(audio, sample_rate, output_file, format="mp3")
            logger.info(f"Generated mp3 file: {output_file.name} in {output_dir}")
        except Exception as e:
            logger.error(f"Error processing {text_name}: {e}")

    # Generate info.json after all audio files have been created
    generate_info_json(resources_dir, output_json)


if __name__ == "__main__":
    # Use click to call main function with args
    sys.argv += ["--resources_dir", "web/resources/leijun"]
    main()