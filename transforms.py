"""
1D augmentations

"""

import numpy as np 
import torch

import random

# for BYOL
class MultiViewDataInjector():
    def __init__(self, transform_list):
        if not isinstance(transform_list, list):
            transform_list = [transform_list]
        self.transform_list = transform_list

    def __call__(self, sample):
        output = [transform(sample).unsqueeze(0) for transform in self.transform_list]
        output_cat = torch.cat(output, dim=0)

        return output_cat




class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self,):
        self.base_transform = BaseTransform()

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k



class RandomResizedCrop(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.n_samples = n_samples

    def forward(self, signal):
        max_samples = signal.shape[-1]
        n_samples = max_samples // random.randint(2, 10)
        start_idx = random.randint(0, max_samples - n_samples)

        signal = signal[..., start_idx : start_idx + n_samples]

        return signal


class Flip(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, signal):
        flipped = torch.flip(signal, [-1])     

        return flipped


class Noise(torch.nn.Module):
    def __init__(self, min_snr=0.0001, max_snr=0.01):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, signal):
        std = torch.std(signal)

        random_factor = torch.rand(1) * (self.max_snr - self.min_snr) + self.min_snr
        random_factor = random_factor.to(std.device)

        # Scale the random_factor by std to get the desired range
        noise_std = random_factor * std

        
        noise = torch.randn(signal.shape).to(noise_std.device) * noise_std # np.random.normal(0.0, noise_std, size=signal.shape).astype(np.float32)

        return signal + noise

class BaseTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.random_noise = RandomApply([Noise()], 0.7)
        self.random_flip = RandomApply([Flip()], 0.4)
        self.compose = torch.nn.Sequential(
            self.random_noise, 
            # RandomResizedCrop()
            
        )
        
    def forward(self, x):
     
        return self.compose(x)
    


class RandomApply(torch.nn.Module):

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, img):
        if self.p < torch.rand(1):
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
    



def get_transforms_list():
    
    return [RandomApply([Noise()], 0.7), \
            RandomApply([Flip()], 0.4) ]