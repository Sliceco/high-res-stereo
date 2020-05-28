import numpy as np
import torch
import torchvision.transforms as transforms

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

__inv_stats = {'mean': [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
               'std': [1 / 0.229, 1 / 0.224, 1 / 0.225]}


class toTensorLegacy(object):
    def __call__(self, pic):
        """
        Args:
            pic (PIL or numpy.ndarray): Image to be converted to tensor

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # This is what TorchVision 0.2.0 returns for transforms.toTensor() for np.ndarray
            return torch.from_numpy(pic.transpose((2, 0, 1))).float().div(255)
        else:
            return transforms.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class FromTensor(object):
    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (pytorch tensor): Image to be converted to tensor

        Returns:
            Tensor: Converted image.
        """
        tensor = tensor.float().mul(255)
        tensor = torch.transpose(tensor, 0, 2)
        tensor = torch.transpose(tensor, 0, 1)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_transform():
    t_list = [
        toTensorLegacy(),
        transforms.Normalize(**__imagenet_stats),
    ]

    return transforms.Compose(t_list)


def get_inv_transform():
    t_list = [
        transforms.Normalize(**__inv_stats),
        FromTensor()
    ]

    return transforms.Compose(t_list)
