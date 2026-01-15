<img src="./mimic-video.png" width="450px"></img>

## Mimic Video (wip)

Implementation of [Mimic-Video](https://mimic-video.github.io/), Video-Action Models for Generalizable Robot Control Beyond VLAs

## Appreciation

- [Pranoy](https://github.com/pranoyr) for submitting a pull request for proprioception masking

## Install

```shell
$ pip install mimic-video
```

## Usage

```python
import torch

# video wrapper
# but will be agnostic to the model

from mimic_video.cosmos_predict import CosmosPredictWrapper

video_wrapper = CosmosPredictWrapper(
    extract_layer = 1,
    random_weights = True,
    tiny = True
)

# mimic video

from mimic_video.mimic_video import MimicVideo

model = MimicVideo(512, video_wrapper)

# states

video = torch.rand(1, 3, 3, 32, 32)

joint_state = torch.randn(1, 32)

# action

actions = torch.randn(1, 32, 20)

# training

loss = model(prompts = '', video = video, actions = actions, joint_state = joint_state)
loss.backward()

# inference

actions = model.sample(prompts = 'pass the butter', video = video, joint_state = joint_state)

assert actions.shape == (1, 32, 20)
```

## Contributing

First make sure `pytest` and test dependencies are installed with

```shell
$ pip install '.[test]'
```

Then add your test to `tests/test_mimic_video.py` and run

```shell
$ pytest tests
```

That's it

## Citations

```bibtex
@inproceedings{Pai2025mimicvideoVM,
    title   = {mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs},
    author  = {Jonas Pai and Liam Achenbach and Victoriano Montesinos and Benedek Forrai and Oier Mees and Elvis Nava},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:283920528}
}
```
