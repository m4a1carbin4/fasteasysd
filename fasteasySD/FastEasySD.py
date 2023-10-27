from .lcm.lcm_scheduler import LCMScheduler
from .lcm.lcm_pipline import LatentConsistencyModelPipeline
from .lcm.lcm_i2i_pipline import LatentConsistencyModelImg2ImgPipeline
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
    """ LCM model pipeline control class.

    Create and manage pipeline objects for LCM models,
    It has functions that process the pipeline input and output values as its main methods.

    """

    def __init__(self, device:str='cpu', use_fp16:bool=False):

        """ Class constructors.

        device : device to use (ex: 'cpu' or 'cuda').

        use_fp16 : Enable fp16 mode. (bool)

        """

        self.user_device = device
        self.user_fp16 = use_fp16
        self.__makeSampler()

    def __makeSampler(self):
        """ Create a txt2img, img2img sampler object. (automatic load with init)

        Create sampler objects for LCM model use.

        """

        self.lcm_sampler = LCM_Sampler()
        self.lcm_i2i_sampler = LCM_img2img_Sampler()
    
    def make_seed(self,seed: int, random_seed:bool) -> int:

        """ Automatically generate seed value (random number)

        Automatically generate seed value (random number)

        seed : user input seed value (int)

        random_seed : True, False for use random_seed
        
        """

        if random_seed:
            seed = random.randint(0,MAX_SEED)
        return seed
    
    def __load_img(self,img_dir):

        """ Load image file for img2img input

        Load the file specified in img_dir and return it to the form available in img2img sampler.

        img_dir : path for input img.

        """

        i = Image.open(img_dir)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return image
    
    def save_PIL(self,pils,save_name):

        """ PIL image list storage function.

        Store a list of PIL images generated by the LCM model.

        pils : list of PIL images

        save_name : Set image save filename. (ex: {save_name}_1.png)

        """

        counter = 0 
        for img in pils:
            img.save(save_name + f"_{counter}.png",compress_level=4)
            counter += 1
    
    def return_PIL(self,images):

        """ Converts LCM Model Tensor results to a PIL list.

        Converts the Tensor list generated through the LCM model to a PIL list.

        images : LCM pipeline output tensor

        """

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
    
    def i2i_batch_save(self,images_list,base_name):

        """ Save img for img2img result values.

        First clean up the Tensor list generated by img2img function.

        and save img2img result.

        images_list : LCM img2img pipeline output tensor list.

        base_name : base name for saving img. (ex : {base_name}_{save_name}_1.png)

        """

        counter = 0
        for images in images_list:
            images = self.return_PIL(images=images)
            self.save_PIL(images,base_name + f"_{counter}")

    def make(self,mode:str,prompt:str,seed:int,steps:int,cfg:int,
                   height:int,width:int,num_images:int,prompt_strength:float=0,input_image_dir:str="./input.jpg"):
        
        """ Process user input and forward it to the LCM pipeline.

        Forward variable values for image creation to the LCM pipeline and return the corresponding result values

        mode : string for LCM mode (txt2img or img2img)

        prompt : LCM model input prompt (ex : "masterpeice, best quality, anime style" )

        seed : seed for LCM model (input 0 will make random seed)

        steps : steps for LCM model (recommend 2~4)

        cfg : cfg for LCM model (recommend 6~8)

        height , width : setting height and width for img (** if you are using img2img w and h should be the same as the input image. **)

        num_images : How many images will you create for this input

        prompt_strength : (only for img2img) How Strong will the prompts be applied in the img2img feature

        input_image_dir : (only for img2img) input image dir

        """
        
        if seed == 0 :
            seed = self.make_seed(0,True)
                
        if mode == "txt2img":
            images = self.lcm_sampler.sample(seed=seed,steps=steps,cfg=cfg,
                             positive_prompt=prompt,height=height,width=width,num_images=num_images,use_fp16=self.user_fp16,device=self.user_device)

        elif mode == "img2img":

            image = self.__load_img(input_image_dir)

            images = self.lcm_i2i_sampler.sample(seed=seed,steps=steps,prompt_strength=prompt_strength,cfg=cfg,images=image,
                             positive_prompt=prompt,height=height,width=width,num_images=num_images,use_fp16=self.user_fp16,device=self.user_device)
            
        else :

            images = None
        return images


    def make_image(self,mode:str,prompt:str,seed:int,steps:int,cfg:int,
                   height:int,width:int,num_images:int,prompt_strength:float=0,input_image_dir:str="./input.jpg",output_image_dir:str="."):
        
        """ Most Simplified Image Generation Function

        Save the image generated by the txt2img, img2img function as a separate file based on user input.

        the output img will be save like output_image_dir/fesd_0.png(txt2img) or output_image_dir/fesd_i2i_0_0.png(img2img)

        mode : string for LCM mode (txt2img or img2img)

        prompt : LCM model input prompt (ex : "masterpeice, best quality, anime style" )

        seed : seed for LCM model (input 0 will make random seed)

        steps : steps for LCM model (recommend 2~4)

        cfg : cfg for LCM model (recommend 6~8)

        height , width : setting height and width for img (** if you are using img2img w and h should be the same as the input image. **)

        num_images : How many images will you create for this input

        prompt_strength : (only for img2img) How Strong will the prompts be applied in the img2img feature

        input_image_dir : (only for img2img) input image dir

        output_image_dir : output image dir (it will not make dir)

        """
        
        images = self.make(mode=mode,prompt=prompt,seed=seed,steps=steps,cfg=cfg,height=height,width=width,num_images=num_images,prompt_strength=prompt_strength,input_image_dir=input_image_dir)

        if mode == "txt2img" and images is not None:
            
            pil_images = self.return_PIL(images)

            self.save_PIL(pils=pil_images,save_name=output_image_dir + "/fesd")

            return True

        elif mode == "img2img" and images is not None:
            
            self.i2i_batch_save(images_list=images,base_name=output_image_dir + "/fesd_i2i")

            return True
        
        else :
            return False