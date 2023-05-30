from dataclasses import dataclass

import yaml

DEFAULT_STEPS = 24
DEFAULT_CFG = 9
DEFAULT_STRENGTH = 0.5


@dataclass
class Config:
    prompt: str
    negative: str
    steps: int
    cfg: float
    strength: float
    seed: int = None
    base_img: str = None
    mask_img: str = None
    model: str = "runwayml/stable-diffusion-v1-5"

    @classmethod
    def sync_from_file(cls):
        # TODO rewrite to sync both as an instance and class
        try:
            with open("conf.yml", "r") as stream:
                conf_raw = yaml.safe_load(stream)
        except FileNotFoundError:
            with open("conf.yml", "w") as stream:
                conf_raw = dict(
                    prompt="",
                    negative="",
                    steps=DEFAULT_STEPS,
                    cfg=DEFAULT_CFG,
                    strength=DEFAULT_STRENGTH,
                    seed=None,
                    mask_img=None,
                    base_img=None,
                )
                yaml.dump(conf_raw, stream, default_flow_style=False)
        return cls(**conf_raw)
