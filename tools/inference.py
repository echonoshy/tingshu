import gc
import os
from pathlib import Path
import numpy as np
import torch
import queue
from loguru import logger
from functools import partial
import soundfile as sf

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
            text=text,
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

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = SpeechInference()
    
    # Generate speech
    try:
        sample_rate, audio = model.generate(
            text="这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。这是一本复盘之书，核心内容来自小米十周年总结。2020年上半年，我和同事们花了大约半年时间，对小米创业历程进行了深入思考与讨论，形成了一系列结论。我的职业生涯经历了30多年的沉浮摔打，从最初学生时代的创业尝试，到开发通用软件、电商、游戏，再到做移动互联网工具、云服务、消费电子硬件、IoT（物联网）智能设备等等，直到去年进入智能电动汽车行业。就像我年轻时听过的鲍勃·迪伦的歌里说的那样，​“答案在风中飘荡”​。一路求索，关于商业思考，不同时期的答案一直在我的脑海中回响飘荡。商业的目的是什么，如何让商业实现最大化的现实意义？我的答案是：效率。它能给最多的人带来最大化的美好幸福感。小米自创立至今12年只干了一件事：用互联网的思维和方法，改造传统制造业，实践、丰富“互联网+制造”​，推动商业社会的效率革命，以实现最大化的用户利益和社会经济运转效率。",
            reference_audio="resources/leijun/leijun.wav",
            reference_text="我当年进的金山，整个公司也只有5-6个人啊，肯定不算什么大厂吧。我觉得进一个好的小厂，和这个小厂一起成长，也是挺好的一种经历。我觉得我每天都在忙着各种有趣的事情，其实眼里有光的人是不会精神内好的。我觉得呢，没有套路就是最好的套路，很多人都说我真诚，其实呢真诚呢让我一路上啊，有不少贵人相助。",
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7
        )
        
        output_path = "gen_speech.wav"
        sf.write(output_path, audio, sample_rate, 'PCM_16')
        print(f"Generated audio with sample rate {sample_rate}Hz and length {len(audio)} samples")
    except Exception as e:
        print(f"Generation failed: {e}")