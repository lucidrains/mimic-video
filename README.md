<img src="./mimic-video.png" width="450px"></img>

## Mimic Video

Implementation of [Mimic-Video](https://mimic-video.github.io/), Video-Action Models for Generalizable Robot Control Beyond VLAs

## Appreciation

- [Pranoy](https://github.com/pranoyr) for submitting a pull request for proprioception masking, fixing the time conditioning of the video model, and Cosmos VAE normalization

- [Pranoy](https://github.com/pranoyr) for a fix to flow calculation for predict clean objective!

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

video = torch.rand(2, 5, 3, 32, 32) # 5 frames, 3 channels, 32 x 32

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

You can also swap out the video backbone for [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3):

```shell
$ pip install "diffusers@git+https://github.com/huggingface/diffusers.git"
```

```python
from mimic_video import MimicVideo, MiniMaxH3PredictWrapper

video_wrapper = MiniMaxH3PredictWrapper(
    model_name = 'MiniMaxAI/MiniMax-H3',
    extract_layer = 34
)

model = MimicVideo(512, video_wrapper)
```

The wrapper handles Qwen3-VL layer-50 hidden states, packed `[text | audio | video]` sequences with 3D MM-RoPE position grids, rectified-flow velocity timesteps, and flow ODE sampling. LoRA finetuning and fixed prefixing work out of the box.

Microphone audio can also be conditioned alongside video:

```python
audio = torch.randn(2, 2, 16000) # (batch, channels, samples)

loss = model(
    prompts = 'pass the butter',
    video = video,
    audio = audio,
    actions = actions,
    joint_state = joint_state
)
```

## LIBERO training

```shell
$ pip install "mimic-video[libero]"
```

Then use the `LiberoDataset`

```python
import torch

from mimic_video import MimicVideo, LiberoDataset
from mimic_video.cosmos_predict import CosmosPredictWrapper

# tiny random-weights video backbone, so this runs anywhere
# swap in a real Cosmos or MiniMax H3 wrapper for actual learning

video_wrapper = CosmosPredictWrapper(
    extract_layer = 1,
    random_weights = True,
    tiny = True
)

# dataset - single test episode

dataset = LiberoDataset('yygx/libero44_KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet')

# mimic video

model = MimicVideo(
    512,
    video_wrapper,
    dim_joint_state = dataset.dim_joint_state,
    dim_action = dataset.dim_action
)

optimizer = torch.optim.Adam(model.action_parameters())

# splat in the batched data

for batch in dataset.get_dataloader(batch_size = 1):
    loss = model(**batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
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

```bibtex
@misc{li2026causalworldmodelingrobot,
    title  = {Causal World Modeling for Robot Control},
    author = {Lin Li and Qihang Zhang and Yiming Luo and Shuai Yang and Ruilin Wang and Fei Han and Mingrui Yu and Zelin Gao and Nan Xue and Xing Zhu and Yujun Shen and Yinghao Xu},
    year   = {2026},
    eprint = {2601.21998},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url    = {https://arxiv.org/abs/2601.21998},
}
```

```bibtex
@misc{kimiteam2026attentionresiduals,
    title   = {Attention Residuals},
    author  = {Kimi Team and Guangyu Chen and Yu Zhang and Jianlin Su and Weixin Xu and Siyuan Pan and Yaoyu Wang and Yucheng Wang and Guanduo Chen and Bohong Yin and Yutian Chen and Junjie Yan and Ming Wei and Y. Zhang and Fanqing Meng and Chao Hong and Xiaotong Xie and Shaowei Liu and Enzhe Lu and Yunpeng Tai and Yanru Chen and Xin Men and Haiqing Guo and Y. Charles and Haoyu Lu and Lin Sui and Jinguo Zhu and Zaida Zhou and Weiran He and Weixiao Huang and Xinran Xu and Yuzhi Wang and Guokun Lai and Yulun Du and Yuxin Wu and Zhilin Yang and Xinyu Zhou},
    year    = {2026},
    eprint  = {2603.15031},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2603.15031},
}
```

```bibtex
@misc{wagenmaker2025steeringdiffusionpolicylatent,
    title   = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
    author  = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
    year    = {2025},
    eprint  = {2506.15799},
    archivePrefix = {arXiv},
    primaryClass = {cs.RO},
    url     = {https://arxiv.org/abs/2506.15799},
}
```

```bibtex
@inproceedings{hendawy2026use,
    title   = {Use the Online Network If You Can: Towards Fast and Stable Reinforcement Learning},
    author  = {Ahmed Hendawy and Henrik Metternich and Th{\'e}o Vincent and Mahdi Kallel and Jan Peters and Carlo D'Eramo},
    booktitle = {The Fourteenth International Conference on Learning Representations},
    year    = {2026},
    url     = {https://openreview.net/forum?id=rFLuaG9Yq6}
}
```

```bibtex
@misc{clavier2024bootstrappingexpectilesreinforcementlearning,
    title   = {Bootstrapping Expectiles in Reinforcement Learning},
    author  = {Pierre Clavier and Emmanuel Rachelson and Erwan Le Pennec and Matthieu Geist},
    year    = {2024},
    eprint  = {2406.04081},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2406.04081},
}
```

```bibtex
@misc{xu2026rltokenbootstrappingonline,
    title   = {RL Token: Bootstrapping Online RL with Vision-Language-Action Models},
    author  = {Charles Xu and Jost Tobias Springenberg and Michael Equi and Ali Amin and Adnan Esmail and Sergey Levine and Liyiming Ke},
    year    = {2026},
    eprint  = {2604.23073},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2604.23073},
}
```

```bibtex
@misc{joseph2026interpretingphysicsvideoworld,
    title   = {Interpreting Physics in Video World Models},
    author  = {Sonia Joseph and Quentin Garrido and Randall Balestriero and Matthew Kowal and Thomas Fel and Shahab Bakhtiari and Blake Richards and Mike Rabbat},
    year    = {2026},
    eprint  = {2602.07050},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url     = {https://arxiv.org/abs/2602.07050},
}
```

```bibtex
@misc{teoh2026nextlatentpredictiontransformerslearn,
    title   = {Next-Latent Prediction Transformers Learn Compact World Models},
    author  = {Jayden Teoh and Manan Tomar and Kwangjun Ahn and Edward S. Hu and Tim Pearce and Pratyusha Sharma and Akshay Krishnamurthy and Riashat Islam and Alex Lamb and John Langford},
    year    = {2026},
    eprint  = {2511.05963},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2511.05963},
}
```

```bibtex
@misc{gladstone2026explorativemodelingunlockingpretraining,
    title   = {Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation},
    author  = {Alexi Gladstone and Heng Ji and Yilun Du},
    year    = {2026},
    eprint  = {2607.27372},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2607.27372},
}
```
