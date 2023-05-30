from PIL import Image
from clip_interrogator import Config, Interrogator
image = Image.open('./example').convert('RGB')
ci = Interrogator(Config(clip_model_name="ViT-L-14/openai",device='cpu'))
print(ci.interrogate(image))
