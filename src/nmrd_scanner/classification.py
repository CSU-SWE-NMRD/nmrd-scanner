import sys
import time
import os
import shutil
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, utils
from torchvision.transforms import v2
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib.pyplot as plt

from nmrd_scanner import metrics
    
_DATA_DIR = 'data/chest_xray'
_MODEL_PATH = 'best_model_params.pt'

# based on https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
_PREPROCESS = {
    'train': v2.Compose([
        v2.ToImage(),
        v2.RGB(),
        v2.RandomResizedCrop(224, scale=(0.9,1.0)),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': v2.Compose([
        v2.ToImage(),
        v2.RGB(),
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

_NUM_EPOCHS = 20

class ClassificationModel:
    def __init__(self):
        self.image_datasets = {}
        self.dataloaders = {}
        self.dataset_sizes = {}

        for x in ['train', 'val']:
            self.image_datasets[x] = datasets.ImageFolder(os.path.join(_DATA_DIR, x), _PREPROCESS[x])
            self.dataloaders[x] = torch.utils.data.DataLoader(self.image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
            self.dataset_sizes[x] = len(self.image_datasets[x])

        self.class_names = self.image_datasets['train'].classes

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device")
        
        self.model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        for param in self.model.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        # Observe that only parameters of final layer are being optimized as
        # opposed to before.
        self.optimizer_conv = optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer_conv, step_size=7, gamma=0.1)
        
        # Corrections for retraining
        self.corrections = {}
    
    def load_trained(self):
        self.model.load_state_dict(torch.load(_MODEL_PATH, weights_only=True, map_location=self.device))
    
    def predict(self, file_path):
        model = self.model
        
        was_training = model.training
        model.eval()

        img = Image.open(file_path).convert("RGB")
        img = _PREPROCESS['val'](img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img.to(self.device))
            probabilities = F.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            model.train(mode=was_training)
            return self.class_names[preds.item()], confidence.item()
        
        model.train(mode=was_training)
        return f"prediction failed", 0.0
    
    def evaluate(self, split='val', save=True):
        model = self.model
    
        was_training = model.training
        model.eval()

        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in self.dataloaders[split]:
                outputs = model(inputs.to(self.device))

                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

        model.train(mode=was_training)
        stats = metrics.compute_metrics(np.concatenate(all_labels), np.concatenate(all_probs), self.class_names)
        if save:
            stats = metrics.save_statistics(stats, split)
        return stats
    
    def train(self):
        model = self.model
        optimizer = self.optimizer_conv
        scheduler = self.exp_lr_scheduler
        dataloaders = self.dataloaders
        device = self.device
        criterion = self.criterion
        dataset_sizes = self.dataset_sizes
        
        since = time.time()
        
        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            temp_model_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(model.state_dict(), temp_model_path)
            best_acc = 0.0

            for epoch in range(_NUM_EPOCHS):
                print(f'Epoch {epoch}/{_NUM_EPOCHS - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        target = labels.data

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics     
                        running_corrects += (preds == target).sum().item()               
                        running_loss += loss.item() * inputs.size(0)
                        
                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects / dataset_sizes[phase]

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        print("using this epoch")
                        torch.save(model.state_dict(), temp_model_path)

                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best accuracy: {best_acc:4f}')

            # load best model weights
            model.load_state_dict(torch.load(temp_model_path, weights_only=True))

            torch.save(model.state_dict(), _MODEL_PATH)

            statistics = self.evaluate(model, 'val')

            print(statistics)

        is_model_ready = True
        model_conv = model
        return statistics
    
    def record_correction(self, file_path, true_label):
        """Keeps track of corrections for later retraining"""
        
        self.corrections[file_path] = true_label.upper()
    
    def retrain(self):
        """Copies images into training dataset and trains the model"""
        
        print('retraining model')
        
        train_dir = self.image_datasets['train'].root
        
        for file_path, label in self.corrections.items():
            file_name = os.path.basename(file_path)
            out_file = train_dir + f'/{label}/{file_name}'
            print(f'Copying {file_path} into {out_file}')
            shutil.copy(file_path, out_file)
        
        self.train()
    
    def visualize(self, num_images=6):
        model = self.model
        dataloaders = self.dataloaders
        device = self.device
        class_names = self.class_names
        
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {class_names[preds[j]]}\nactual: {class_names[labels[j]]}')
                    _imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)
            
def main():
    model = ClassificationModel()
    
    should_train = input('train? y/n: ')

    if should_train == 'y':
        should_multithread = input('use multithreading? y/n: ')
        if should_multithread == 'y':
            for x in ['train', 'val']:
                model.dataloaders[x] = torch.utils.data.DataLoader(
                    model.image_datasets[x], batch_size=4, shuffle=True, num_workers=4)

        model.train()
    else:
        model.load_trained()
        print(model.evaluate('val'))
    
    model.visualize()
    plt.ioff()
    plt.show()

    input("press enter to quit")
    sys.exit()

def _imshow(inp, title=None):
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
