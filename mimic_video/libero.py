from __future__ import annotations

import os
import re
from glob import glob
from itertools import groupby
from collections import namedtuple

import torch
from einops import rearrange
from torch import stack, tensor
from torch.nn.functional import interpolate
from torch.utils.data import Dataset, DataLoader

from mimic_video.utils import exists, check_import

# constants

DEFAULT_LIBERO_REPO = 'yygx/libero44_KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet'

Episode = namedtuple('Episode', ['video_path', 'states', 'actions', 'instruction'])

# helper functions

def clean_instruction(raw):
    raw = str(raw)

    match = re.search(r"b'([^']*)'", raw)
    return match.group(1) if exists(match) else raw

def libero_collate(batch):
    videos, joint_states, actions, instructions = zip(*batch)

    return dict(
        video = stack(videos),
        joint_state = stack(joint_states),
        actions = stack(actions),
        prompts = list(instructions)
    )

# dataset

class LiberoDataset(Dataset):
    def __init__(
        self,
        repo_id = DEFAULT_LIBERO_REPO,
        *,
        num_frames = 5,
        action_chunk_len = 32,
        frame_skip = 1,
        resize = 64
    ):
        super().__init__()

        check_import('av', error_message = '`av` must be installed - `pip install "mimic-video[libero]"`')
        check_import('pyarrow', error_message = '`pyarrow` must be installed - `pip install "mimic-video[libero]"`')
        check_import('huggingface_hub', error_message = '`huggingface_hub` must be installed - `pip install "mimic-video[libero]"`')

        from huggingface_hub import snapshot_download

        self.num_frames = num_frames
        self.action_chunk_len = action_chunk_len
        self.frame_skip = frame_skip
        self.resize = resize

        self._video_cache = {}

        self.root = snapshot_download(
            repo_id,
            repo_type = 'dataset',
            allow_patterns = ['data/*.parquet', 'data/**/*.parquet', 'videos/**/*', 'meta_data/*']
        )

        self._load_episodes()

    def _load_episodes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_paths = glob(os.path.join(self.root, 'data', '**', '*.parquet'), recursive = True)
        table = pa.concat_tables([pq.read_table(path) for path in parquet_paths])

        images = table.column('observation.images.image').to_pylist()
        states = table.column('observation.state').to_pylist()
        actions = table.column('action').to_pylist()
        instructions = table.column('language_instruction').to_pylist()
        episode_ids = table.column('episode_index').to_pylist()

        grouped_rows = groupby(zip(images, states, actions, instructions, episode_ids), key = lambda t: t[-1])

        self.episodes = []

        for _, episode_rows in grouped_rows:
            episode_rows = list(episode_rows)

            first_image = episode_rows[0][0]
            assert isinstance(first_image, dict), 'only LeRobot v3 datasets, with separate video files, are supported'

            video_path = os.path.join(self.root, first_image['path'])

            episode_states = tensor([row[1] for row in episode_rows]).float()
            episode_actions = tensor([row[2] for row in episode_rows]).float()

            self.episodes.append(Episode(video_path, episode_states, episode_actions, clean_instruction(episode_rows[0][3])))

        # dimensions for the model

        self.dim_joint_state = self.episodes[0].states.shape[-1]
        self.dim_action = self.episodes[0].actions.shape[-1]

        # all valid chunk positions, so the sampled window fits within the episode

        window = max(self.action_chunk_len, (self.num_frames - 1) * self.frame_skip + 1)

        self.indices = []

        for episode_idx, episode in enumerate(self.episodes):
            num_states = episode.states.shape[0]
            self.indices.extend(
                (episode_idx, start) for start in range(num_states - window + 1)
            )

        assert len(self.indices) > 0, f'all episodes are shorter than the window of {window} frames'

    def __len__(self):
        return len(self.indices)

    def _video_frames(self, episode, start):
        import av

        path = episode.video_path

        if path not in self._video_cache:
            frames = []

            with av.open(path) as container:
                for frame in container.decode(video = 0):
                    frame = frame.to_ndarray(format = 'rgb24')
                    frames.append(torch.from_numpy(frame).float() / 255.)

            num_states = episode.states.shape[0]
            frames = stack(frames)[:num_states]
            frames = rearrange(frames, 't h w c -> t c h w')

            self._video_cache[path] = frames

        stop = start + (self.num_frames - 1) * self.frame_skip + 1
        frames = self._video_cache[path][start : stop : self.frame_skip]

        if not exists(self.resize):
            return frames

        return interpolate(frames, size = (self.resize, self.resize), mode = 'bilinear', antialias = True)

    def __getitem__(self, idx):
        episode_idx, start = self.indices[idx]
        episode = self.episodes[episode_idx]

        video = self._video_frames(episode, start)
        joint_state = episode.states[start]
        actions = episode.actions[start : start + self.action_chunk_len]

        return video, joint_state, actions, episode.instruction

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn = libero_collate, **kwargs)
