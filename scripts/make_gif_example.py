import imageio
images = []
from os import walk
from PIL import Image
from cv2 import addWeighted
SUBSTRING = "568676694"
PATH = './example'
import numpy as np


filenames = []
for (dirpath, dirnames, filenames) in walk(PATH):
    filenames.extend(filenames)
    break

filenames = sorted(list(set(filenames)))

def blend(list_images): # Blend images equally.
    equal_fraction = 1.0 / (len(list_images))
    output = np.zeros_like(list_images[0])
    for img in list_images:
        output = output + img * equal_fraction
    output = output.astype(np.uint8)
    return output

for filename in filenames:
    if SUBSTRING in filename:
        img = imageio.v2.imread(f'{PATH}/{filename}')
        if len(images)>0:
            images.append(addWeighted(images[-1], 0.5, img, 0.5, 0))
        images.append(img)

imageio.mimsave('movie.gif', images, duration=100)
