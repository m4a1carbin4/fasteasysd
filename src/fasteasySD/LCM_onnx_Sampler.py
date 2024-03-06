import torch

from diffusers import LCMScheduler
from optimum.onnxruntime import ORTStableDiffusionPipeline, ORTStableDiffusionXLPipeline

from os import path
import time
import numpy as np

class LCM_onnx_Sampler:
    def __init__(self):
        self.pipe = None

    def make_pipeline(self, model_path, model_type):
        
        if model_type in ["SDXL","SSD-1B"]:
            pipe = ORTStableDiffusionXLPipeline.from_pretrained(
                model_path
            )
        else :
            pipe = ORTStableDiffusionPipeline.from_pretrained(
                model_path
            )
            
        return pipe

    def sample(self, model_path, lora_path, lora_name, model_type, seed, steps, cfg, positive_prompt, negative_prompt, prompt_strength, height, width, num_images, use_fp16, device):
        if self.pipe is None:
            self.pipe = self.make_pipeline(model_path=model_path,model_type=model_type)
            
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

            self.pipe.to(device=device)

        torch.manual_seed(seed)
        start_time = time.time()

        result = self.pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            output_type="np",
        ).images

        print("LCM inference time: ", time.time() - start_time, "seconds")
        images_tensor = torch.from_numpy(result)

        return (images_tensor,)