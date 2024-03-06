import torch

from diffusers import LCMScheduler, LatentConsistencyModelImg2ImgPipeline

from os import path
import time
import numpy as np

class LCM_img2img_Sampler:
    def __init__(self):
        self.pipe = None

    def sample(self, model_path, lora_path, lora_name, model_type, seed, steps, cfg, images, positive_prompt, negative_prompt, prompt_strength, height, width, num_images, use_fp16, device):
        if self.pipe is None:
            self.pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
                safety_checker=None,
            )
            
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

            if use_fp16:
                self.pipe.to(torch_device=device,
                             torch_dtype=torch.float16)
            else:
                self.pipe.to(torch_device=device,
                             torch_dtype=torch.float32)

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
                width=width,
                height=height,
                guidance_scale=cfg,
                num_inference_steps=steps,
                num_images_per_prompt=num_images,
                lcm_origin_steps=50,
                output_type="np",
                ).images
            tensor_results = [torch.from_numpy(np_result) for np_result in result]
            results.extend(tensor_results)

        results = torch.stack(results)
        
        print("LCM img2img inference time: ", time.time() - start_time, "seconds")

        return (results,)