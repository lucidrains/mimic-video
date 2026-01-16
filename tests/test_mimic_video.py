import pytest
param = pytest.mark.parametrize

import torch

@param('num_residual_streams', (1, 4))
def test_mimic_video(
    num_residual_streams
):
    from mimic_video.mimic_video import MimicVideo

    video_hiddens = torch.randn(2, 64, 77)
    video_mask = torch.randint(0, 2, (2, 64)).bool()

    mimic_video = MimicVideo(512, dim_video_hidden = 77, num_residual_streams = num_residual_streams)

    actions = torch.randn(2, 32, 20)

    joint_state = torch.randn(2, 32)

    forward_kwargs = dict(video_hiddens = video_hiddens, context_mask = video_mask, joint_state = joint_state)

    loss = mimic_video(actions = actions, **forward_kwargs)

    assert loss.numel() == 1

    flow = mimic_video(actions = actions, **forward_kwargs, time = torch.tensor([0.5, 0.5]))

    assert flow.shape == actions.shape

@param('num_residual_streams', (1, 4))
def test_e2e(
    num_residual_streams
):
    from mimic_video.mimic_video import MimicVideo
    from mimic_video.cosmos_predict import CosmosPredictWrapper

    video_wrapper = CosmosPredictWrapper(
        extract_layer = 1,
        random_weights = True,
        tiny = True
    )

    model = MimicVideo(512, video_wrapper, num_residual_streams = num_residual_streams)

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

    pred_actions = model.sample(video = video, joint_state = joint_state, prompts = 'pass the butter')
    assert pred_actions.shape == (1, 32, 20)
