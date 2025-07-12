# from diffusers import StableDiffusionPipeline
# from realesrgan import RealESRGAN_x4plus
# from PIL import Image
# import torch

# # # Load SD model
# # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
# # pipe.enable_xformers_memory_efficient_attention()

# # # Generate image
# # prompt = "a detailed painting of a futuristic cityscape at dusk"
# # image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
# 

# # Upscale with Real-ESRGAN
# model = RealESRGAN_x4plus(device=torch.device("mps"))
# model.load_weights('RealESRGAN_x4.pth')  # You must download this checkpoint manually

# upscaled = model.predict(image)

# # Save
# upscaled.save("upscaled_output.png")

from PIL import Image
import numpy as np
import torch
import cv2
from realesrgan import RealESRGANer

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

from basicsr.archs.rrdbnet_arch import RRDBNet

model_arch = RRDBNet(num_in_ch=3, num_out_ch=3)


model = RealESRGANer(device=device, scale=4, model_path='scripts/RealESRGAN_x4plus.pth', model=model_arch)

loadnet = torch.load('scripts/RealESRGAN_x4plus.pth', map_location=device)

def upscale_image(pil_img: Image.Image) -> Image.Image:
    image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    sr_image, _ = model.enhance(image_bgr, outscale=2)
    print("After ESRGAN:", sr_image.shape)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(sr_image)

image = Image.open('in.png').convert('RGB')
print(image.size)
image_out = upscale_image(image)
image_out = image_out.resize((1024, 1024), Image.LANCZOS)

image_out.save('out.png')
