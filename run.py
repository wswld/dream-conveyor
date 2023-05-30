from src.conveyor import DreamConveyor
from src.enums import Gradient

import click

@click.command()
@click.option('--gradient', '-g', default=0)
@click.argument('path')
def go_br(gradient, path: str):
    conveyor = DreamConveyor(
        model='runwayml/stable-diffusion-v1-5',
        workspace_path=path,
        gradient=Gradient(gradient),
        local=False,
        controlnet=True
    )
    conveyor.go_br(prompt_prepend=None)

if __name__ == '__main__':
    go_br()
