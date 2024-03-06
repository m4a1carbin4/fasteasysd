import torch

from diffusers import LCMScheduler, LatentConsistencyModelPipeline
import time

class LCM_Sampler:
    def __init__(self):
        self.pipe = None
    
    def sample(self, model_path, lora_path, lora_name, model_type, seed, steps, cfg, prompt_strength, positive_prompt, negative_prompt, height, width, num_images, use_fp16, device):
        if self.pipe is None:
            self.pipe = LatentConsistencyModelPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
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

        result = self.pipe(
            prompt=positive_prompt,
            width=width,
            height=height,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            lcm_origin_steps=50,
            output_type="np",
        ).images

        print("LCM inference time: ", time.time() - start_time, "seconds")
        images_tensor = torch.from_numpy(result)

        return (images_tensor,)