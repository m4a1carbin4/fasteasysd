import torch

from diffusers import LCMScheduler
from optimum.onnxruntime import ORTStableDiffusionImg2ImgPipeline, ORTStableDiffusionXLImg2ImgPipeline

from os import path
import time
import numpy as np

class LCM_onnx_img2img_Sampler:
    def __init__(self):
        self.pipe = None

    def make_pipeline(self, model_path, model_type):
        
        if model_type in ["SDXL","SSD-1B"]:
            pipe = ORTStableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_path
            )
        else :
            pipe = ORTStableDiffusionImg2ImgPipeline.from_pretrained(
                model_path
            )
            
        return pipe

    def sample(self, model_path, lora_path, lora_name, model_type, seed, steps, cfg, images, positive_prompt, negative_prompt, prompt_strength, height, width, num_images, use_fp16, device):
        if self.pipe is None:
            self.pipe = self.make_pipeline(model_path=model_path,model_type=model_type)
            
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        torch.manual_seed(seed)
        start_time = time.time()

        images = np.transpose(images, (0, 3, 1, 2))
        results = []
        for i in range(images.shape[0]):
            image = images[i]
            result = self.pipe(
                image=image,
                prompt=positive_prompt,
                strength=prompt_strength,
                guidance_scale=cfg,
                num_inference_steps=steps,
                num_images_per_prompt=num_images,
                output_type="np",
                ).images
            tensor_results = [torch.from_numpy(np_result) for np_result in result]
            results.extend(tensor_results)

        results = torch.stack(results)
        
        print("LCM_onnx_img2img inference time: ", time.time() - start_time, "seconds")

        return (results,)