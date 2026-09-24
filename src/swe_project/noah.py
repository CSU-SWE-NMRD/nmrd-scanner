import sys
import os
import torch
from PIL import Image
from torchvision import datasets, models, utils
from torchvision.transforms import v2
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'data/chest_xray'

def main():
    # based on https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # TODO: different transformations for training?
    preprocess = v2.Compose([
        v2.ToImage(),
        v2.RGB(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    
    for x in ['train', 'val']:
        image_datasets[x] = datasets.ImageFolder(os.path.join(data_dir, x), preprocess)
        dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
        dataset_sizes[x] = len(image_datasets[x])
    
    class_names = image_datasets['train'].classes
    
    # TODO: GPU acceleration
    
    def imshow(inp, title=None):
        """Display image for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
    
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])
    input('')
    
    sys.exit()
