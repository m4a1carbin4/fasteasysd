from fasteasySD import fesd

infer = fesd(device='cuda',use_fp16=False,mode="SD")

infer_result = infer.make(model_path="/home/waganawa/stable-diffusion-webui/models/Stable-diffusion/milkyWonderland_v20.safetensors",
                lora_path="/home/waganawa/Documents/Code/Python_Project/project_lcm_python/chamchamAICPU/models/",lora_name="chamcham_new_train_lora_2-000001.safetensors",
                prompt="masterpiece, best quality, chamcham(twitch), hair bell, hair ribbon, multicolored hair, two-tone hair, 1girl, solo,",
                n_prompt="bad hand,text,watermark,low quality,medium quality,blurry,censored,wrinkles,deformed,mutated text,watermark,low quality,medium quality,blurry,censored,wrinkles,deformed,mutated",
                seed=0,steps=8,cfg=2,height=960,width=512,prompt_strength=0.4,num_images=1)

pil_images = infer.return_PIL(infer_result)

for img in pil_images :
    img.save("test.png")