from src.conveyor import DreamConveyor
from unittest.mock import patch, PropertyMock
from src.enums import Gradient
from src.config import Config


DEFAULT_TEXT2IMG_CONFIG = Config(
    prompt='', 
    negative='', 
    steps=20,
    cfg=9,
    strength=1, 
    seed=None, 
    base_img=None, 
    mask_img=None, 
)


@patch('src.conveyor.DiffusionPipeline')
@patch('src.conveyor.EulerAncestralDiscreteScheduler')
@patch('PIL.Image')
def test_simple_txt2img_run(img, scheduler, pipe):
    conveyor = DreamConveyor(
        model='test/test',
        workspace_path='test/',
        gradient=Gradient.NoGradient,
        local=False,
        controlnet=False,
        scheduler=scheduler
    )
    with patch.object(DreamConveyor, 'conf', new_callable=PropertyMock) as mocked_conf:
        mocked_conf.return_value = DEFAULT_TEXT2IMG_CONFIG
        conveyor.go_br(prompt_prepend=None)
        assert pipe.from_pretrained()().images.__getitem__().save.call_count == 100
        for c in pipe.from_pretrained().call_args_list:
            if c.kwargs:
                assert c.kwargs['num_inference_steps'] == 20
                assert c.kwargs['guidance_scale'] == 9
