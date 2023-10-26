from lcm.lcm_scheduler import LCMScheduler
from lcm.lcm_pipline import LatentConsistencyModelPipeline
from lcm.lcm_i2i_pipline import LatentConsistencyModelImg2ImgPipeline
from torchvision import transforms

from os import path
import time
import torch
import random
import numpy as np

from PIL import Image,ImageOps

MAX_SEED = np.iinfo(np.int32).max

def make_seed(seed: int, random_seed:bool) -> int:
    if random_seed:
        seed = random.randint(0,MAX_SEED)
    return seed

class LCM_Sampler:
    def __init__(self):
        self.scheduler = LCMScheduler.from_pretrained(
            path.join(path.dirname(__file__), "scheduler_config.json"))
        self.pipe = None
    
    def sample(self, seed, steps, cfg, positive_prompt, height, width, num_images, use_fp16,device):
        if self.pipe is None:
            self.pipe = LatentConsistencyModelPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
                scheduler=self.scheduler,
                safety_checker=None,
            )

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
    
class LCM_img2img_Sampler:
    def __init__(self):
        self.scheduler = LCMScheduler.from_pretrained(
            path.join(path.dirname(__file__), "scheduler_config.json"))
        self.pipe = None

    def sample(self, seed, steps, prompt_strength, cfg, images, positive_prompt, height, width, num_images, use_fp16, device):
        if self.pipe is None:
            self.pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
                safety_checker=None,
            )

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

class FastEasySD:
    def __init__(self, device:str, use_fp16:bool):

        self.user_device = device
        self.user_fp16 = use_fp16
        self.convert_tensor = transforms.ToTensor()
        self.__makeSampler()

    def __makeSampler(self):
        self.lcm_sampler = LCM_Sampler()
        self.lcm_i2i_sampler = LCM_img2img_Sampler()
    
    def __make_seed(self,seed: int, random_seed:bool) -> int:
        if random_seed:
            seed = random.randint(0,MAX_SEED)
        return seed
    
    def __load_img(self,img_dir):
        i = Image.open(img_dir)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return image
    
    def __save_PIL(self,pils,save_name):

        counter = 0 
        for img in pils:
            img.save(save_name + f"_{counter}.png",compress_level=4)
            counter += 1
    
    def __return_PIL(self,images):
        images = images[0]

        if images.ndim == 3:
            images = images.numpy()[None, ...]
        else :
            images = images.numpy()
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images
    
    def __i2i_batch_save(self,images_list,base_name):

        counter = 0
        for images in images_list:
            images = self.__return_PIL(images=images)
            self.__save_PIL(images,base_name + f"_{counter}")

    def make_image(self,mode:str,prompt:str,seed:int,steps:int,cfg:int,
                   height:int,width:int,num_images:int,prompt_strength:float=0,input_image_dir:str="./input.jpg",output_image_dir:str="."):
        if seed == 0 :
            seed = self.__make_seed(0,True)

        if mode == "txt2img":
            images = self.lcm_sampler.sample(seed=seed,steps=steps,cfg=cfg,
                             positive_prompt=prompt,height=height,width=width,num_images=num_images,use_fp16=self.user_fp16,device=self.user_device)
            
            pil_images = self.__return_PIL(images)

            self.__save_PIL(pils=pil_images,save_name=output_image_dir + "/fesd")

        elif mode == "img2img":

            image = self.__load_img(input_image_dir)

            images = self.lcm_i2i_sampler.sample(seed=seed,steps=steps,prompt_strength=prompt_strength,cfg=cfg,images=image,
                             positive_prompt=prompt,height=height,width=width,num_images=num_images,use_fp16=self.user_fp16,device=self.user_device)
            
            self.__i2i_batch_save(images_list=images,base_name=output_image_dir + "/fesd_i2i")

test = FastEasySD(device='cpu',use_fp16=False)

test.make_image(mode="img2img",prompt="masterpeice, best quality, anime style",seed=0,steps=4,cfg=8,height=1063,width=827,num_images=1,prompt_strength=0.5,input_image_dir="input.jpg")

test.make_image(mode="txt2img",prompt="masterpeice, best quality, anime style",seed=0,steps=4,cfg=8,height=768,width=512,num_images=2)