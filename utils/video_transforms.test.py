import torch
import numpy as np
from video_transforms import NoiseTransforms, ShuffleTransforms

# Test ShuffleTransforms
def test_shuffle_transforms():
    # Test shuffle frames
    shuffle_transforms = ShuffleTransforms(mode='T')
    clip = torch.rand(3, 16, 224, 224)
    shuffled_clip = shuffle_transforms(clip)
    assert shuffled_clip.shape == clip.shape
    assert not np.allclose(shuffled_clip.numpy(), clip.numpy(), atol=1e-3)

    # Test shuffle channels
    shuffle_transforms = ShuffleTransforms(mode='C')
    clip = torch.rand(3, 16, 224, 224)
    shuffled_clip = shuffle_transforms(clip)
    assert shuffled_clip.shape == clip.shape
    assert not np.allclose(shuffled_clip.numpy(), clip.numpy(), atol=1e-3)

    # Test shuffle width
    shuffle_transforms = ShuffleTransforms(mode='W')
    clip = torch.rand(3, 16, 224, 224)
    shuffled_clip = shuffle_transforms(clip)
    assert shuffled_clip.shape == clip.shape
    assert not np.allclose(shuffled_clip.numpy(), clip.numpy(), atol=1e-3)

    # Test shuffle height
    shuffle_transforms = ShuffleTransforms(mode='H')
    clip = torch.rand(3, 16, 224, 224)
    shuffled_clip = shuffle_transforms(clip)
    assert shuffled_clip.shape == clip.shape
    assert not np.allclose(shuffled_clip.numpy(), clip.numpy(), atol=1e-3)

    # Test shuffle all
    shuffle_transforms = ShuffleTransforms(mode='TCHW')
    clip = torch.rand(3, 16, 224, 224)
    shuffled_clip = shuffle_transforms(clip)
    assert shuffled_clip.shape == clip.shape
    assert not np.allclose(shuffled_clip.numpy(), clip.numpy(), atol=1e-3)

test_noise_transforms()
test_shuffle_transforms()