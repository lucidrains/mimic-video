<img src="./mimic-video.png" width="450px"></img>

## Mimic Video

Implementation of [Mimic-Video](https://mimic-video.github.io/), Video-Action Models for Generalizable Robot Control Beyond VLAs

## Appreciation

- [Pranoy](https://github.com/pranoyr) for submitting a pull request for proprioception masking as well as fixing the time conditioning of the video model

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
@misc{li2025basicsletdenoisinggenerative,
    title   = {Back to Basics: Let Denoising Generative Models Denoise}, 
    author  = {Tianhong Li and Kaiming He},
    year    = {2025},
    eprint  = {2511.13720},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url     = {https://arxiv.org/abs/2511.13720}, 
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

```bibtex
@misc{intelligence2025pi06vlalearnsexperience,
    title   = {$\pi^{*}_{0.6}$: a VLA That Learns From Experience}, 
    author  = {Physical Intelligence and Ali Amin and Raichelle Aniceto and Ashwin Balakrishna and Kevin Black and Ken Conley and Grace Connors and James Darpinian and Karan Dhabalia and Jared DiCarlo and Danny Driess and Michael Equi and Adnan Esmail and Yunhao Fang and Chelsea Finn and Catherine Glossop and Thomas Godden and Ivan Goryachev and Lachy Groom and Hunter Hancock and Karol Hausman and Gashon Hussein and Brian Ichter and Szymon Jakubczak and Rowan Jen and Tim Jones and Ben Katz and Liyiming Ke and Chandra Kuchi and Marinda Lamb and Devin LeBlanc and Sergey Levine and Adrian Li-Bell and Yao Lu and Vishnu Mano and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Allen Z. Ren and Charvi Sharma and Lucy Xiaoyang Shi and Laura Smith and Jost Tobias Springenberg and Kyle Stachowicz and Will Stoeckle and Alex Swerdlow and James Tanner and Marcel Torne and Quan Vuong and Anna Walling and Haohuan Wang and Blake Williams and Sukwon Yoo and Lili Yu and Ury Zhilinsky and Zhiyuan Zhou},
    year    = {2025},
    eprint  = {2511.14759},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2511.14759}, 
}
```

```bibtex
@misc{kim2026cosmospolicyfinetuningvideo,
    title   = {Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning}, 
    author  = {Moo Jin Kim and Yihuai Gao and Tsung-Yi Lin and Yen-Chen Lin and Yunhao Ge and Grace Lam and Percy Liang and Shuran Song and Ming-Yu Liu and Chelsea Finn and Jinwei Gu},
    year    = {2026},
    eprint  = {2601.16163},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI},
    url     = {https://arxiv.org/abs/2601.16163}, 
}
```
