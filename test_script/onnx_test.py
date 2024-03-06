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

    def sample(self, model_path, lora_path, lora_name, model_type, seed, steps, cfg, images, positive_prompt, negative_prompt, prompt_strength, height, width, num_images, use_fp16, device):
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
    
test_infer = LCM_onnx_Sampler().sample(model_path="WGNW/chamcham_v1_checkpoint_onnx",
                                       lora_path="/home/waganawa/Documents/Code/Python_Project/project_lcm_python/chamchamAICPU/models/",
                                       lora_name="chamcham_new_train_lora_2-000001.safetensors",
                                       model_type="SD",
                                       positive_prompt="masterpiece, best quality, chamcham(twitch), hair bell, hair ribbon, multicolored hair, two-tone hair, 1girl, solo,",
                                       negative_prompt="nsfw,nude,bad hand,text,watermark,low quality,medium quality,bad quality",
                                       seed=0,
                                       steps=8,
                                       cfg=2.0,
                                       height=512,
                                       width=512,
                                       use_fp16=False,
                                       device='cpu',
                                       images=None,
                                       prompt_strength=1,
                                       num_images=1
                                       )