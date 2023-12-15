# This code is based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
import torch
import torch.nn.functional as F
import numpy as np


class DiffAug():
    def __init__(self,
                 strategy='color_crop_cutout_flip_scale_rotate',
                 batch=False,
                 ratio_cutout=0.5,
                 single=False):
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = ratio_cutout
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5
        self.batch = batch

        self.aug = True
        if strategy == '' or strategy.lower() == 'none':
            self.aug = False
        else:
            self.strategy = []
            self.flip = False
            self.color = False
            self.cutout = False
            for aug in strategy.lower().split('_'):
                if aug == 'flip' and not single:
                    self.flip = True
                elif aug == 'color' and not single:
                    self.color = True
                elif aug == 'cutout' and not single:
                    self.cutout = True
                else:
                    self.strategy.append(aug)

        self.aug_fn = {
            'color': [self.brightness_fn, self.saturation_fn, self.contrast_fn],
            'crop': [self.crop_fn],
            'cutout': [self.cutout_fn],
            'flip': [self.flip_fn],
            'scale': [self.scale_fn],
            'rotate': [self.rotate_fn],
            'translate': [self.translate_fn],
        }

def __call__(self, x, single_aug=True, seed=-1):
    if not self.aug:
        return x
    else:
        if self.flip:
            self.set_seed(seed)
            x = self.flip_fn(x, self.batch)  # ensure this function is PyTorch-compatible
        if self.color:
            for f in self.aug_fn['color']:
                self.set_seed(seed)
                x = f(x, self.batch)  # ensure this function is PyTorch-compatible
    if len(self.strategy) > 0:
            if single_aug:
                # single
                idx = torch.randint(0, len(self.strategy), (1,)).item()  # replace np with torch
                p = self.strategy[idx]
                for f in self.aug_fn[p]:
                    self.set_seed(seed)
                    x = f(x, self.batch)  # ensure this function is PyTorch-compatible
                else:
                # multiple
                    for p in self.strategy:
                     for f in self.aug_fn[p]:
                        self.set_seed(seed)
                        x = f(x, self.batch)  # ensure this function is PyTorch-compatible
    if self.cutout:
        self.set_seed(seed)
        x = self.cutout_fn(x, self.batch)  # ensure this function is PyTorch-compatible
    return x.contiguous()

def set_seed(self, seed):
    if seed != -1:
        torch.manual_seed(seed)

def scale_fn(self, x, batch=True):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = self.ratio_scale

    if batch:
        sx = torch.rand(1, device=x.device) * (ratio - 1.0 / ratio) + 1.0 / ratio
        sy = torch.rand(1, device=x.device) * (ratio - 1.0 / ratio) + 1.0 / ratio
        theta = torch.tensor([[sx.item(), 0, 0], [0, sy.item(), 0]], dtype=torch.float, device=x.device)
        theta = theta.expand(x.shape[0], 2, 3)
    else:
        sx = torch.rand(x.shape[0], device=x.device) * (ratio - 1.0 / ratio) + 1.0 / ratio
        sy = torch.rand(x.shape[0], device=x.device) * (ratio - 1.0 / ratio) + 1.0 / ratio
        theta = torch.stack([[sx[i], 0, 0], [0, sy[i], 0]] for i in range(x.shape[0]))

    grid = F.affine_grid(theta, x.shape)
    x = F.grid_sample(x, grid)
    return x

def rotate_fn(self, x, batch=True):
    # [-180, 180], 90: anticlockwise 90 degree
    ratio = self.ratio_rotate

    if batch:
        theta = (torch.rand(1, device=x.device) - 0.5) * 2 * ratio / 180 * torch.tensor([np.pi], device=x.device)
        cos_theta = torch.cos(theta).item()
        sin_theta = torch.sin(theta).item()
        theta = torch.tensor([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0]], dtype=torch.float, device=x.device)
        theta = theta.expand(x.shape[0], 2, 3)
    else:
        theta_vals = (torch.rand(x.shape[0], device=x.device) - 0.5) * 2 * ratio / 180 * torch.tensor([np.pi], device=x.device)
        theta = torch.stack([[
            torch.cos(theta_vals[i]), -torch.sin(theta_vals[i]), 0], 
            [torch.sin(theta_vals[i]), torch.cos(theta_vals[i]), 0]
        ] for i in range(x.shape[0]))

    grid = F.affine_grid(theta, x.shape)
    x = F.grid_sample(x, grid)
    return x

def flip_fn(self, x, batch=True):
        prob = self.prob_flip

        if batch:
            coin = torch.rand(1, device=x.device).item()
            if coin < prob:
                return x.flip(3)
            else:
                return x
        else:
            randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            return torch.where(randf < prob, x.flip(3), x)

def brightness_fn(self, x, batch=True):
        ratio = self.brightness

        if batch:
            randb = torch.rand(1, device=x.device).item()
        else:
            randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = x + (randb - 0.5) * ratio
        return x

def saturation_fn(self, x, batch=True):
        ratio = self.saturation

        x_mean = x.mean(dim=1, keepdim=True)
        if batch:
            rands = torch.rand(1, device=x.device).item()
        else:
            rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x

def contrast_fn(self, x, batch=True):
        ratio = self.contrast

        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        if batch:
            randc = torch.rand(1, device=x.device).item()
        else:
            randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x
def translate_fn(self, x, batch=True):
        ratio = self.ratio_crop_pad

        shift_y = int(x.size(3) * ratio + 0.5)
        if batch:
            translation_y = np.random.randint(-shift_y, shift_y + 1)
        else:
            translation_y = torch.randint(-shift_y,
                                          shift_y + 1,
                                          size=[x.size(0), 1, 1],
                                          device=x.device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, (1, 1))
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

def crop_fn(self, x, batch=True):
        # The image is padded on its surrounding and then cropped.
        ratio = self.ratio_crop_pad

        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        if batch:
            translation_x = np.random.randint(-shift_x, shift_x + 1)
            translation_y = np.random.randint(-shift_y, shift_y + 1)
        else:
            translation_x = torch.randint(-shift_x,
                                          shift_x + 1,
                                          size=[x.size(0), 1, 1],
                                          device=x.device)

            translation_y = torch.randint(-shift_y,
                                          shift_y + 1,
                                          size=[x.size(0), 1, 1],
                                          device=x.device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, (1, 1, 1, 1))
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

def cutout_fn(self, x, batch=True):
    ratio = self.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

    if batch:
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), (1,)).item()
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), (1,)).item()
    else:
        offset_x = torch.randint(0,
                                 x.size(2) + (1 - cutout_size[0] % 2),
                                 size=[x.size(0), 1, 1],
                                 device=x.device)

        offset_y = torch.randint(0,
                                 x.size(3) + (1 - cutout_size[1] % 2),
                                 size=[x.size(0), 1, 1],
                                 device=x.device)

    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

def cutout_inv_fn(self, x, batch=True):
    ratio = self.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

    if batch:
        offset_x = torch.randint(0, x.size(2) - cutout_size[0], (1,)).item()
        offset_y = torch.randint(0, x.size(3) - cutout_size[1], (1,)).item()
    else:
        offset_x = torch.randint(0,
                                 x.size(2) - cutout_size[0],
                                 size=[x.size(0), 1, 1],
                                 device=x.device)
        offset_y = torch.randint(0,
                                 x.size(3) - cutout_size[1],
                                 size=[x.size(0), 1, 1],
                                 device=x.device)

    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y, min=0, max=x.size(3) - 1)
    mask = torch.zeros(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 1.
    x = x * mask.unsqueeze(1)
    return x