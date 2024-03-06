from fasteasySD import fesd

cpu_infer = fesd(device='cpu',use_fp16=False,mode="ONNX")

cpu_infer.make_image(model_path="WGNW/chamcham_v1_checkpoint_onnx",
                     prompt="masterpiece, best quality, chamcham(twitch), hair bell, hair ribbon, multicolored hair, two-tone hair, 1girl, solo,",
                     n_prompt="nsfw,nude,bad hand,text,watermark,low quality,medium quality,bad quality",
                     steps=8,cfg=2.5,width=512,height=512)

cpu_img2img_infer = fesd(device='cpu',use_fp16=False,mode="ONNX",img2img=True)

cpu_img2img_infer.make_image(model_path="WGNW/chamcham_v1_checkpoint_onnx",
                     input_image_dir="201835543.jpg",
                     prompt="masterpiece, best quality,anime style",
                     n_prompt="nsfw,nude,bad hand,text,watermark,low quality,medium quality,bad quality",
                     steps=8,cfg=2.5,width=512,height=512,prompt_strength=0.3)

