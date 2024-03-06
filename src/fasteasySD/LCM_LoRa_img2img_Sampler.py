import torch

from diffusers import LCMScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline

from os import path
import time
import numpy as np

class LCM_LoRa_img2img_Sampler:
    def __init__(self):
        self.scheduler = None
        self.pipe = None

    def make_pipeline(self, model_path, lora_path, lora_name, use_fp16, model_type):

        adapter_id = None
        if model_type == "SD":
            adapter_id = "latent-consistency/lcm-lora-sdv1-5"
        elif model_type == "SDXL":
            adapter_id = "latent-consistency/lcm-lora-sdxl"
        elif model_type == "SSD-1B":
            adapter_id = "latent-consistency/lcm-lora-ssd-1b"

        if path.isfile(path.abspath(model_path)):

            if model_type in ["SDXL","SSD-1B"]:

                if use_fp16 : 
                    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                        model_path,torch_dtype=torch.float16, use_safetensors=True
                    )
                else :
                    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                        model_path,torch_dtype=torch.float32, use_safetensors=True
                    )
            else :
                
                if use_fp16 : 
                    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                        model_path,torch_dtype=torch.float16, use_safetensors=True
                    )
                else :
                    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                        model_path,torch_dtype=torch.float32, use_safetensors=True
                    )

        else:
        
            if model_type in ["SDXL","SSD-1B"]:

                if use_fp16 : 
                    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        model_path,torch_dtype=torch.float16, use_safetensors=True
                    )
                else :
                    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        model_path,torch_dtype=torch.float32, use_safetensors=True
                    )
            else :
                
                if use_fp16 : 
                    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        model_path,torch_dtype=torch.float16, use_safetensors=True
                    )
                else :
                    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        model_path,torch_dtype=torch.float32, use_safetensors=True
                    )
        
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

        if lora_path is not None:

            if lora_name is not None :
                pipe.load_lora_weights(lora_path,weight_name=lora_name)
                pipe.fuse_lora(lora_scale=1)
            else :
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora(lora_scale=1)

        pipe.load_lora_weights(adapter_id)
            
        return pipe

    def sample(self, model_path, lora_path, lora_name, model_type, seed, steps, cfg, images, positive_prompt, negative_prompt, prompt_strength, height, width, num_images, use_fp16, device):
        if self.pipe is None :
            self.pipe = self.make_pipeline(model_path=model_path, lora_path=lora_path, lora_name=lora_name, use_fp16=use_fp16, model_type=model_type)

            self.pipe.to(device)
            
            #self.pipe.enable_vae_slicing()
            #self.pipe.enable_vae_tiling()
            #self.pipe.enable_xformers_memory_efficient_attention()

        torch.manual_seed(seed)
        start_time = time.time()
        
        images = np.transpose(images, (0, 3, 1, 2))
        results = []
        for i in range(images.shape[0]):
            image = images[i]
            result = self.pipe(
                image=image,
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                strength=prompt_strength,
                width=width,
                height=height,
                guidance_scale=cfg,
                num_inference_steps=steps,
                num_images_per_prompt=num_images,
                output_type="np",
                ).images
            tensor_results = [torch.from_numpy(np_result) for np_result in result]
            results.extend(tensor_results)

        results = torch.stack(results)
        
        print(f"{model_type} img2img inference time: ", time.time() - start_time, "seconds")
        
        return (results,)