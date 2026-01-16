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

from mimic_video import MimicVideo

model = MimicVideo(512, video_wrapper)

# states

video = torch.rand(2, 3, 3, 32, 32)

joint_state = torch.randn(2, 32)

# action

actions = torch.randn(2, 32, 20)

# training

loss = model(
    prompts = [
        'put the package on the conveyer belt',
        'pass the butter'
    ],
    video = video,
    actions = actions,
    joint_state = joint_state
)

loss.backward()

# inference

actions = model.sample(
    prompts = 'peel the orange',
    video = video[:1],
    joint_state = joint_state[:1]
)

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

```bibtex
@misc{black2025trainingtimeactionconditioningefficient,
    title   = {Training-Time Action Conditioning for Efficient Real-Time Chunking}, 
    author  = {Kevin Black and Allen Z. Ren and Michael Equi and Sergey Levine},
    year    = {2025},
    eprint  = {2512.05964},
    archivePrefix = {arXiv},
    primaryClass = {cs.RO},
    url     = {https://arxiv.org/abs/2512.05964}, 
}
```

