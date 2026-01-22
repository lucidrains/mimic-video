import pytest
param = pytest.mark.parametrize

import torch

@param('model_output_clean', (False, True))
@param('num_residual_streams', (1, 4))
@param('train_time_rtc', (False, True))
@param('action_stats_given', (False, True))
@param('condition_tokens_given', (False, True))
def test_mimic_video(
    model_output_clean,
    num_residual_streams,
    train_time_rtc,
    action_stats_given,
    condition_tokens_given,
):
    from mimic_video.mimic_video import MimicVideo

    video_hiddens = torch.randn(2, 64, 77)
    video_mask = torch.randint(0, 2, (2, 64)).bool()

    action_mean_std = None
    if action_stats_given:
        action_mean_std = torch.ones((2, 20))

    advantage_ids = task_ids = None
    if condition_tokens_given:
        advantage_ids = torch.randint(0, 2, (2,))
        task_ids = torch.randint(0, 3, (2,))

    mimic_video = MimicVideo(
        512,
        action_mean_std = action_mean_std,
        dim_video_hidden = 77,
        num_residual_streams = num_residual_streams,
        train_time_rtc = train_time_rtc,
        train_time_rtc_max_delay = 4,
        num_advantage_ids = 2,
        num_task_ids = 3,
        model_output_clean = model_output_clean
    )

    actions = torch.randn(2, 32, 20)

    joint_state = torch.randn(2, 32)

    forward_kwargs = dict(video_hiddens = video_hiddens, context_mask = video_mask, joint_state = joint_state, advantage_ids = advantage_ids, task_ids = task_ids)

    loss = mimic_video(actions = actions, **forward_kwargs)

    assert loss.numel() == 1

    flow = mimic_video(actions = actions, **forward_kwargs, time = torch.tensor([0.5, 0.5]))

    assert flow.shape == actions.shape

@param('num_residual_streams', (1, 4))
@param('prev_action_chunk', (False, True))
@param('cross_attend_multiple', (False, True))
def test_e2e(
    num_residual_streams,
    prev_action_chunk,
    cross_attend_multiple
):
    from mimic_video.mimic_video import MimicVideo
    from mimic_video.cosmos_predict import CosmosPredictWrapper

    if cross_attend_multiple:
        extract_layer = [19, 20]
        extracted_video_layer_indices = [0, 1, 1] # first layer attends to 19, second and third to 20
    else:
        extract_layer = 19
        extracted_video_layer_indices = None

    video_wrapper = CosmosPredictWrapper(
        extract_layer = extract_layer,
        random_weights = True,
        tiny = True,
    )

    model = MimicVideo(
        512,
        video_wrapper,
        num_residual_streams = num_residual_streams,
        depth = 3,
        extracted_video_layer_indices = extracted_video_layer_indices
    )

    video = torch.rand(1, 3, 3, 32, 32)

    actions = torch.randn(1, 32, 20)

    joint_state = torch.randn(1, 32)

    loss = model(
        video = video,
        actions = actions,
        joint_state = joint_state,
        prompts = 'put the package on the conveyer belt'
    )

    loss.backward()

    prefix_action_chunk = None
    if prev_action_chunk:
        prefix_action_chunk = torch.randn(1, 4, 20)

    pred_actions = model.sample(
        video = video,
        joint_state = joint_state,
        prompts = 'pass the butter',
        prefix_action_chunk = prefix_action_chunk
    )

    assert pred_actions.shape == (1, 32, 20)

def test_lora_e2e():
    import os
    import shutil
    from torch.utils.data import Dataset
    from mimic_video.mimic_video import MimicVideo
    from mimic_video.cosmos_predict import CosmosPredictWrapper

    class DummyRobotDataset(Dataset):
        def __len__(self): return 1
        def __getitem__(self, _):
            return torch.rand(9, 3, 32, 32), torch.randint(0, 1000, (32,))

    save_path = './cosmos-lora-test'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    # 1. setup video wrapper and finetune to get a lora

    video_wrapper = CosmosPredictWrapper(
        extract_layers = [1, 2],
        random_weights = True,
        tiny = True
    )

    video_wrapper.finetune(
        DummyRobotDataset(),
        save_path = save_path,
        epochs = 1
    )

    # 2. instantiate new wrapper with the trained lora_path

    lora_wrapper = CosmosPredictWrapper(
        extract_layers = [1, 2],
        random_weights = True,
        tiny = True,
        lora_path = save_path
    )

    # 3. mimic video integration

    model = MimicVideo(
        dim = 512,
        video_predict_wrapper = lora_wrapper,
        depth = 3,
        extracted_video_layer_indices = [0, 1, 1]
    )

    # 4. dummy states and actions

    video = torch.rand(1, 9, 3, 32, 32)
    joint_state = torch.randn(1, 32)
    actions = torch.randn(1, 32, 20)

    # 5. training forward pass

    loss = model(
        prompts = 'a task',
        video = video,
        actions = actions,
        joint_state = joint_state
    )

    loss.backward()
    assert loss.numel() == 1

    # 6. inference sampling

    sampled_actions = model.sample(
        prompts = 'the final task',
        video = video,
        joint_state = joint_state
    )

    assert sampled_actions.shape == (1, 32, 20)

    # cleanup

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
