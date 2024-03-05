import torch
from torchvision.transforms import Compose
import numpy as np

class NoiseTransforms:
    def __init__(self, noise_type='gaussian', mean=0.0, std=1.0):
        self.noise_type = noise_type
        self.mean = mean
        self.std = std

class ShuffleTransforms:
    """
    A video transformation class that shuffles the frames, channels, width, or height of a video clip.

    Args:
        mode (str): A string indicating the mode of shuffling. Can contain any combination of the following characters:
            - 'T': Shuffle the frames of the video clip.
            - 'C': Shuffle the channels of the video clip.
            - 'W': Shuffle the width of the video clip.
            - 'H': Shuffle the height of the video clip.
    """

    def __init__(self, mode="CWH"):
        self.mode = mode

    def __call__(self, clip):
        assert len(clip.shape) == 4, "clip should be a 4D tensor."
        C, T, W, H = clip.shape
        np.random.seed(3407)
        if "C" in self.mode:
            index = np.arange(C)
            np.random.shuffle(index)
            clip = clip[index, :, :, :]
        if "T" in self.mode:
            index = np.arange(T)
            np.random.shuffle(index)
            clip = clip[:, index, :, :]
        if "W" in self.mode:
            index = np.arange(W)
            np.random.shuffle(index)
            clip = clip[:, :, index, :]
        if "H" in self.mode:
            index = np.arange(H)
            np.random.shuffle(index)
            clip = clip[:, :, :, index]

        return clip


