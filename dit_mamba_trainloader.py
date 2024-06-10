import torch
import os
import datetime
import time
from IPython.display import clear_output
from pathlib import Path
from collections import namedtuple
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

Sample = namedtuple("Sample", ("im", "noisy_im", "noise_level"))

def show(image):
    
    reverse_transforms = transforms.Compose([
        #transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.detach().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if not isinstance(image, torch.Tensor) or image.ndim == 4:
        image = torch.cat(tuple(image), -1)
    display(reverse_transforms(image.cpu()))

class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    channels = 3
    image_size = 64
    shape = (channels, image_size, image_size)
    dataset = "cats"


def alpha_blend(a, b, alpha):
    return alpha * a + (1 - alpha) * b


def to_device(ims):
    return Sample(*(x.to(Config.device) for x in ims))

@torch.no_grad()
def generate_images(model, n_images=16, n_steps=100, step_size=2.0):
    model.eval()
    x, prev = torch.rand(n_images, *Config.shape, device=Config.device), None
    noise_levels = torch.linspace(1, 0, n_steps + 1, device=Config.device)
    for nl_in, nl_out in zip(noise_levels, noise_levels[1:]):
        denoised = pred = model(x, nl_in.view(1, 1, 1, 1)).denoised.clamp(0,1)
        if prev is not None: denoised = prev + step_size * (denoised - prev)
        x, prev = alpha_blend(x, denoised, nl_out / nl_in), pred
    model.train()
    return x.clamp(0, 1)



class Visualizer:
    def __init__(self):
        self.smoothed_loss = None
        self.losses_since_last_vis = []
        self.avg_losses = []
        self.steps = []
        self.step = 0
        self.t_last_vis = 0
        self.t_last_save = 0
        self.t_start = None
        folder, idx = datetime.datetime.now().strftime("%Y_%m_%d") + "_training_logs", 0
        while Path(f"{folder}_{idx}").exists():
            idx += 1
        self.folder = Path(f"{folder}_{idx}")
        self.folder.mkdir()
    def __call__(self, model, t, x, y, loss, n_demo=16):
        self.losses_since_last_vis.append(loss)
        self.smoothed_loss = loss if self.smoothed_loss is None else 0.99 * self.smoothed_loss + 0.01 * loss
        self.step += 1
        if self.t_start is None:
            self.t_start = t
        if t > self.t_last_vis + 30:
            generated_images = generate_images(model, n_images=n_demo)
            clear_output(wait=True)
            print("Input Noisified Image, Noise Level")
            show(x.noisy_im[:n_demo])
            show(x.noise_level.expand(len(x.noise_level), 3, 16, Config.image_size)[:n_demo])
            print("Predictions")
            show(y.denoised[:n_demo].clamp(0, 1))
            print("Targets")
            show(x.im[:n_demo])
            self.steps.append(self.step)
            self.avg_losses.append(sum(self.losses_since_last_vis) / len(self.losses_since_last_vis))
            self.losses_since_last_vis = []
            print("Generated Images (Averaged Model)")
            show(generated_images)
            plt.title("Losses")
            plt.plot(self.steps, self.avg_losses)
            plt.gcf().set_size_inches(16, 4)
            plt.ylim(0, 1.5 * self.avg_losses[-1])
            if t > self.t_last_save + 120:
                torch.save(model.state_dict(), self.folder / "model.pth")
                torch.save((self.steps, self.avg_losses), self.folder / "stats.pth")
                TF.to_pil_image(torch.cat(tuple(generated_images), -1)).save(self.folder / f"generated_{self.step:07d}.jpg", quality=95)
                plt.gcf().savefig(self.folder / "stats.jpg")
                self.t_last_save = t
            plt.show()
            self.t_last_vis = t
        print(
            f"\r{self.step: 5d} Steps; {int(t - self.t_start): 3d} Seconds; "
            f"{60 * self.step / (t - self.t_start + 1):.1f} Steps / Min; "
            f"{len(x.im) * 60 * self.step / (t - self.t_start + 1):.1f} Images / Min; "
            f"Smoothed Loss {self.smoothed_loss:.5f}; "
        , end="")
        
class Looper(torch.utils.data.Dataset):
    def __init__(self, dataset, n=1<<20):
        self.dataset = dataset
        self.n = n
    def __len__(self):
        return max(len(self.dataset), self.n)
    def __getitem__(self, i):
        return self.dataset[i % len(self.dataset)]

from torchvision.models import vgg16
from torch.nn.functional import l1_loss, mse_loss

class PerceptualLoss(nn.Module):
    def __init__(self, x):
        super(PerceptualLoss, self).__init__()
        # Using VGG16 pre-trained model
        vgg = models.vgg16(pretrained=True).features[:x]  # Using only the first few layers up to the third maxpooling layer
        # Freeze all VGG parameters since we're only using it for feature extraction
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        vgg.to(Config.device)

    def forward(self, input, target):
        # Normalize input and target images to fit VGG's expected parameter ranges
        input = self.normalize_batch(input)
        target = self.normalize_batch(target)
        # Extract features
        features_input = self.vgg(input)
        features_target = self.vgg(target)
        # Compute perceptual loss
        loss = l1_loss(features_input, features_target)
        return loss

    def normalize_batch(self, batch):
        # Normalize using imagenet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        batch = (batch - mean) / std
        return batch

perceptual_loss3 = PerceptualLoss(3)
perceptual_loss8 = PerceptualLoss(8)

class Trainer:
    def __init__(self, model, avg_model, dataset, batch_size=16, learning_rate=1e-4):
        self.model = model
        self.avg_model = avg_model
        self.last_avg_time = time.time()
        self.opt = torch.optim.AdamW(model.parameters(), learning_rate, amsgrad=True)
        num_workers = min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
        self.dataloader = torch.utils.data.DataLoader(Looper(dataset), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        self.dl_iter = iter(self.dataloader)
        self.visualizer = Visualizer()

    def avg_model_step(self, t):
        if t > self.last_avg_time + 2:
            self.avg_model.update_parameters(self.model)
            self.last_avg_time = t

    def get_batch(self):
        try:
            batch = next(self.dl_iter)
        except StopIteration:
            self.dl_iter = iter(self.dataloader)
            batch = next(self.dl_iter)
        return to_device(batch)

    def train(self, n_seconds):
        self.model.train()
        start_time = time.time()
        while time.time() < start_time + n_seconds:
            self.train_step(time.time())

    def train_step(self, t):
        x = self.get_batch()
        y = self.model(x.noisy_im, x.noise_level)
        loss = perceptual_loss8(y.denoised, x.im) + mse_loss(y.denoised, x.im)

        self.opt.zero_grad(); loss.backward(); self.opt.step(); self.avg_model_step(t)
        self.visualizer(self.avg_model, t, x, y, loss.item())
