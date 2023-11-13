import torch

from diffusers import LCMScheduler, LatentConsistencyModelPipeline, LatentConsistencyModelImg2ImgPipeline, StableDiffusionPipeline

from os import path
import time
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
    
class LoRa_Sampler:
    def __init__(self):
        self.scheduler = None
        self.pipe = None

    def make_pipeline(self, model_path, lora_path, lora_name, use_fp16, model_type):

        if path.isfile(path.abspath(model_path)):

            if use_fp16 : 
                pipe = StableDiffusionPipeline.from_single_file(
                    model_path,torch_dtype=torch.float16, use_safetensors=True
                )
            else :
                pipe = StableDiffusionPipeline.from_single_file(
                    model_path,torch_dtype=torch.float32, use_safetensors=True
                )

        else:
            
            if use_fp16 : 
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,torch_dtype=torch.float16, use_safetensors=True
                )
            else :
                pipe = StableDiffusionPipeline.from_pretrained(
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
        
            adapter_id = None
            if model_type == "SD":
                adapter_id = "latent-consistency/lcm-lora-sdv1-5"
            elif model_type == "SDXL":
                adapter_id = "latent-consistency/lcm-lora-sdxl"
            elif model_type == "SSD-1B":
                adapter_id = "latent-consistency/lcm-lora-ssd-1b"

            pipe.load_lora_weights(adapter_id)
            
        return pipe

    def sample(self, model_path, lora_path, lora_name, model_type, seed, steps, cfg, positive_prompt, negative_prompt, height, width, use_fp16, device):
        if self.pipe is None :
            self.pipe = self.make_pipeline(model_path=model_path, lora_path=lora_path, lora_name=lora_name, use_fp16=use_fp16, model_type=model_type)

            self.pipe.to(device)

        torch.manual_seed(seed)
        start_time = time.time()

        result = self.pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=cfg,
            num_inference_steps=steps,
            output_type="np",
            cross_attention_kwargs={"scale": 0.8}
        ).images

        print("LCM inference time: ", time.time() - start_time, "seconds")
        images_tensor = torch.from_numpy(result)

        return (images_tensor,)

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
        self.lora_sampler = LoRa_Sampler()
    
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

    def make(self,mode:str,model_path:str,model_type:str,lora_path:str,lora_name:str,**kwargs):
        
        """ Process user input and forward it to the LCM pipeline.

        Forward variable values for image creation to the LCM pipeline and return the corresponding result values

        mode : string for LCM mode (txt2img or img2img)
        
        model_path : path for model (huggingface repo id or path str)
        
        model_type : type of model ("LCM","SD","SDXL","SSD-1B")
        
        lora_path : path of lora file (ex : "./path/for/lora")
        
        lora_name : name for lora file (ex : "test.safetensor")
        
        input_image_dir : (only for img2img) input image dir

        output_image_dir : output image dir (it will not make dir)

        prompt : model input prompt (ex : "masterpeice, best quality, anime style" )
        
        n_prompt : model negative input prompt (ex : "nsfw,nude,bad quality" )

        seed : seed for LCM model (input 0 will make random seed)

        steps : steps for LCM model (recommend 2~4)

        cfg : cfg for LCM model (recommend 6~8)

        height , width : setting height and width for img (** if you are using img2img w and h should be the same as the input image. **)

        num_images : How many images will you create for this input

        prompt_strength : (only for img2img) How Strong will the prompts be applied in the img2img feature

        """
        if "input_image_dir" in kwargs:
            input_image_dir = kwargs.get("input_image_dir")
        else:
            input_image_dir = "./input.jpg"
        
        if "prompt" in kwargs:
            prompt = kwargs.get("prompt")
        else :
            prompt = "masterpiece"
        if "n_prompt" in kwargs:
            n_prompt = kwargs.get("n_prompt")
        else :
            n_prompt = "nsfw, nude, worst quality, low quality"
        if "seed" in kwargs:
            seed = kwargs.get("seed")
        else : 
            seed = 0
        if "steps" in kwargs:
            steps = kwargs.get("steps")
        else :
            steps = 4
        if "cfg" in kwargs:
            cfg = kwargs.get("cfg")
        else :
            cfg = 2
        if "width" in kwargs:
            width = kwargs.get("width")
        else :
            width = 512
        if "height" in kwargs:
            height = kwargs.get("height")
        else :
            height = 512
        if "num_images" in kwargs:
            num_images = kwargs.get("num_images")
        else :
            num_images = 1
        if "prompt_strength" in kwargs:
            prompt_strength = kwargs.get("prompt_strength")
        else :
            prompt_strength = 0.5
        
        if seed == 0 :
            seed = self.make_seed(0,True)
                
        if mode == "txt2img":
            if model_type == "LCM":
                images = self.lcm_sampler.sample(seed=seed,steps=steps,cfg=cfg,
                             positive_prompt=prompt,height=height,width=width,num_images=num_images,use_fp16=self.user_fp16,device=self.user_device)
            elif model_type in ["SD","SDXL","SSD-1B"]:
                images = self.lora_sampler.sample(model_path=model_path, lora_path=lora_path, lora_name=lora_name, model_type=model_type, seed=seed,steps=steps,cfg=cfg,
                             positive_prompt=prompt, negative_prompt=n_prompt, height=height,width=width,use_fp16=self.user_fp16,device=self.user_device)
        elif mode == "img2img":

            image = self.__load_img(input_image_dir)

            images = self.lcm_i2i_sampler.sample(seed=seed,steps=steps,prompt_strength=prompt_strength,cfg=cfg,images=image,
                             positive_prompt=prompt,height=height,width=width,num_images=num_images,use_fp16=self.user_fp16,device=self.user_device)
            
        else :

            images = None
        return images


    def make_image(self,mode:str,model_path:str=None,model_type:str="LCM",lora_path:str=None,lora_name:str=None,output_image_dir:str=".",**kwargs):
        
        """ Most Simplified Image Generation Function

        Save the image generated by the txt2img, img2img function as a separate file based on user input.

        the output img will be save like output_image_dir/fesd_0.png(txt2img) or output_image_dir/fesd_i2i_0_0.png(img2img)

        mode : string for LCM mode (txt2img or img2img)
        
        model_path : path for model (huggingface repo id or path str)
        
        model_type : type of model ("LCM","SD","SDXL","SSD-1B")
        
        lora_path : path of lora file (ex : "./path/for/lora")
        
        lora_name : name for lora file (ex : "test.safetensor")
        
        input_image_dir : (only for img2img) input image dir

        output_image_dir : output image dir (it will not make dir)

        prompt : model input prompt (ex : "masterpeice, best quality, anime style" )
        
        n_prompt : model negative input prompt (ex : "nsfw,nude,bad quality" )

        seed : seed for LCM model (input 0 will make random seed)

        steps : steps for LCM model (recommend 2~4)

        cfg : cfg for LCM model (recommend 6~8)

        height , width : setting height and width for img (** if you are using img2img w and h should be the same as the input image. **)

        num_images : How many images will you create for this input

        prompt_strength : (only for img2img) How Strong will the prompts be applied in the img2img feature

        """
        
        images = self.make(mode=mode,model_path=model_path,model_type=model_type,lora_path=lora_path,lora_name=lora_name,**kwargs)

        if mode == "txt2img" and images is not None:
            
            pil_images = self.return_PIL(images)

            self.save_PIL(pils=pil_images,save_name=output_image_dir + "/fesd")

            return True

        elif mode == "img2img" and images is not None:
            
            self.i2i_batch_save(images_list=images,base_name=output_image_dir + "/fesd_i2i")

            return True
        
        else :
            return False
        
test = FastEasySD(device='cuda',use_fp16=True)

test.make_image(mode="txt2img",
                model_type="SD",model_path="milkyWonderland_v20.safetensors",
                lora_path=".",lora_name="chamcham_new_train_lora_2-000001.safetensors",
                prompt="sharp details, sharp focus, anime style, masterpiece, best quality, chamcham(twitch), hair bell, hair ribbon, multicolored hair, two-tone hair, 1girl, solo, orange shirt, long hair, hair clip",
                n_prompt="noisy, blurry, grainy,text, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, (realistic, lip, nose, tooth, rouge, lipstick, eyeshadow:1.0),black and white, low contrast",
                seed=0,steps=8,cfg=2,height=960,width=512,num_images=1)

"""test.make_image(mode="img2img",
                model_type="LCM",
                prompt="sharp details, sharp focus, glasses, anime style, 1man",
                seed=0,steps=8,cfg=2,height=960,width=512,num_images=1,prompt_strength=0.8,input_image_dir="input.jpg")"""