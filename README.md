# dream-conveyor

This is a framework on top of a framework to fascilitate (and automate) 
generation of large quantities of ungodly visual horrors and occasionall AI art. 
Sometimes the latter look more like the former, never vice versa. Use at your 
own risk.

## Rationale

When initially generating AI art using my CPU (I'm cheap), waiting for 
individual results is time-consuming. A background script based on the 
[diffusers library](https://github.com/huggingface/diffusers) by Hugging Face 
would simplify the process. Initially, I wrote custom code for various tasks, 
but it lacked scalability and reusability. Over time, I identified recurring 
usage patterns and developed a framework to generate images efficiently in the 
background. Especially when parameters are changed slightly with each iteration 
(see [Gradients](#Gradients)).

**DISCLAIMER**: This piece of software is a result of me implementing _my_ 
personal take on Stable Diffusion image generation workflow. If you find it 
useful — great, otherwise you should probably look elsewhere. 

## Quickstart

Start by installing `poetry` globally, it will help you manage the rest of the 
requirements and create a virtual environment for you.

``` sh
pip install poetry
```

Now install the dependencies by running the following in the project folder.

``` sh
poetry install
```

After poetry install you will probably be switched to a virtual environment 
(i.e poetry shell) automatically. If not, run `poetry shell`.

The next step would be to configure `accelerate`.

``` sh
accelerate config
```

In some more complicated cases that might not be enough, refer to 
[accelerate documentation](https://huggingface.co/docs/accelerate/basic_tutorials/install) 
if it's giving you issues.

Most of this can be run w/o `accelerate` but it does accelerate things 
immensely.

Now you're ready to start generating images. Run this command to start:

``` sh
accelerate launch run.py ./example/
```

`./example/` can be replaced with any folder on your machine. I have several 
folders on mine that I use for different projects.

When the script is run it will create the folder (if it doesn't exist) and 
add a configuration file in that folder that will probably look like this:

``` yaml
base_img: null
cfg: 9
mask_img: null
negative: ''
prompt: ''
seed: null
steps: 24
strength: 0.5
```

**You can update this configuration even while the script is running**, that's 
the whole point. If you change any of the parameters, they will take effect 
for the next image that will start generation.

I will cover all the implications and various scenarios in series of more 
in-depth tutorials.

## Gradients

TBD

## Supported Models

Tested with the following models:

- dreamlike-art/dreamlike-photoreal-2.0
- wavymulder/Analog-Diffusion
- runwayml/stable-diffusion-v1-5
- runwayml/stable-diffusion-inpainting
- stabilityai/stable-diffusion-2
- stabilityai/stable-diffusion-2-inpainting
- darkstorm2150/Protogen_x3.4_Official_Release
