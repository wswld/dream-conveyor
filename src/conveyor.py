import os
from random import randint
from time import time
from typing import Optional

import torch
from diffusers import (DiffusionPipeline, StableDiffusionXLInpaintPipeline, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler,
                       SchedulerMixin, StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline,
                       ControlNetModel)
from diffusers import UniPCMultistepScheduler
from PIL import Image, PngImagePlugin

from src.config import Config
from src.utils import print_params_in_color

from .enums import RANGES, Device, Gradient

DEFAULT_TENSOR_SIZE = 1024


class DreamConveyor:
    def __init__(
        self,
        model: str,
        workspace_path: str,
        gradient: Gradient = Gradient.NoGradient,
        device: Device = Device.MPS,
        scheduler: SchedulerMixin = EulerAncestralDiscreteScheduler,
        local: bool = True,
        controlnet: bool = True
    ) -> None:
        self.model_name = model
        self.scheduler = scheduler
        self.workspace = self._mkdir_cd(workspace_path)
        self.gradient = gradient
        self.device = device
        self.generator = torch.Generator(device=self.device.value)
        self._pipeline = None
        self._conf = None
        self.base_image = None
        self.local = local
        self._mask = None
        self.controlnet = controlnet

    @property
    def conf(self) -> Config:
        self._conf = Config.sync_from_file()
        return self._conf

    @property
    def image(self) -> Config:
        conf = self.conf
        if conf.base_img:
            init_image = Image.open(conf.base_img).convert("RGB")
            init_image.thumbnail((DEFAULT_TENSOR_SIZE, DEFAULT_TENSOR_SIZE))
            self.base_image = init_image
        else:
            self.base_image = None
        return self.base_image

    @property
    def mask(self) -> Config:
        conf = self.conf
        if conf.mask_img:
            mask = Image.open(conf.mask_img)
            mask.thumbnail((DEFAULT_TENSOR_SIZE, DEFAULT_TENSOR_SIZE))
            self._mask = mask
        else:
            self._mask = None
        return self._mask

    @property
    def pipeline(self):
        conf = self.conf
        if (
            conf.base_img
            and conf.mask_img
            and (
                not self._pipeline
                or not isinstance(self._pipeline, StableDiffusionXLInpaintPipeline)
            )
        ):
            self._pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.model_name, local_files_only=self.local
            )
            self._scheduler_and_safety()
        print("PIPELINE BEING REUSED?", isinstance(self._pipeline, StableDiffusionXLImg2ImgPipeline), self._pipeline is not None, self.model_name)
        if (
            conf.base_img
            and not conf.mask_img
            and (
                not self._pipeline
                or not isinstance(self._pipeline, StableDiffusionXLImg2ImgPipeline)
            )
        ):
            print('here')
            if self.controlnet:
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-scribble")
                self._pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    self.model_name, controlnet=controlnet, local_files_only=self.local
                )
            else:
                self._pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.model_name, local_files_only=self.local
                )
                self._scheduler_and_safety()
        if not conf.base_img and (
            not self._pipeline
            or isinstance(self._pipeline, StableDiffusionXLImg2ImgPipeline)
        ):
            self._pipeline = DiffusionPipeline.from_pretrained(
                self.model_name, local_files_only=self.local
            )
            self._scheduler_and_safety()
            self.base_image = None
        return self._pipeline

    def _gimme_seed(self):
        return randint(42, 4294967295)

    def _scheduler_and_safety(self):
        if self.controlnet:
            self._pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self._pipeline.scheduler.config)
        else:
            self._pipeline.scheduler = self.scheduler.from_config(
                self._pipeline.scheduler.config
            )
        self._pipeline.to(self.device.value)

        # ¯\_(ツ)_/¯
        # safety checkers although useful result in false positive even on SFW imagery
        def dummy(images, **kwargs):
            return images, False

        self._pipeline.safety_checker = dummy

    def _reset_scheduler(self, steps: int):
        scheduler_config = self._pipeline.scheduler.config
        scheduler_config['num_train_timesteps'] = steps
        self._pipeline.scheduler = self.scheduler.from_config(scheduler_config)

    def _mkdir_cd(self, workspace_path: str) -> str:
        if not os.path.exists(workspace_path):
            os.makedirs(workspace_path)
        os.chdir(workspace_path)
        return workspace_path

    def go_br(self, prompt_prepend: Optional[list] = None):
        r = RANGES[self.gradient] if not prompt_prepend else len(
            prompt_prepend)
        for i in range(r):
            # torch.mps.empty_cache()
            conf = self.conf
            add_params = {"strength": conf.strength}
            image = self.image
            mask = self.mask
            if image is not None:
                add_params["image"] = image
            if mask is not None:
                add_params["mask_image"] = self.mask
                add_params.pop("strength")
            if image is None or self.controlnet:
                add_params.pop("strength")
            seed = conf.seed if conf.seed else randint(42, 4294967295)
            self.generator.manual_seed(seed)
            cfg = conf.cfg if not self.gradient == Gradient.CFGGradient else i * 0.5
            if self.gradient == Gradient.StrengthGradient:
                add_params["strength"] = (i + 1) / 10
            conf.steps = i + 1 if self.gradient == Gradient.StepsGradient else conf.steps
            print_params_in_color(
                seed,
                cfg,
                conf.steps,
                conf.strength,
                conf.base_img,
                conf.mask_img,
                conf.model,
            )
            if prompt_prepend:
                final_prompt = f"{prompt_prepend[i]}, {conf.prompt}"
            else:
                final_prompt = conf.prompt
            image = self.pipeline(
                prompt=final_prompt,
                negative_prompt=conf.negative,
                num_inference_steps=conf.steps,
                guidance_scale=cfg,
                generator=self.generator,
                **add_params,
            ).images[0]
            scheduler_config = self._pipeline.scheduler.config
            print(scheduler_config['num_train_timesteps'])
            step = i if not prompt_prepend else prompt_prepend[i].replace(
                " ", "")
            print(self.pipeline.text_encoder.config.model_type)
            # Create metadata dict
            meta = PngImagePlugin.PngInfo()
            meta.add_text("Prompt", final_prompt)
            meta.add_text("Seed", str(seed))
            meta.add_text("CFG", str(cfg))
            meta.add_text("Steps", str(conf.steps))
            meta.add_text("Model", self.model_name)

            image.save(
                f"{str(int(time()))}-cfg{cfg}-steps{conf.steps}-seed{seed}.png", pnginfo=meta)
