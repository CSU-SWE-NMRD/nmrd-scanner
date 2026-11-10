import sys
import time
import os
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

data_dir = 'data/chest_xray'
model_path = 'best_model_params.pt'

# based on https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
preprocess = {
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

image_datasets = {}
dataloaders = {}
dataset_sizes = {}

for x in ['train', 'val']:
    image_datasets[x] = datasets.ImageFolder(os.path.join(data_dir, x), preprocess[x])
    dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
    dataset_sizes[x] = len(image_datasets[x])

class_names = image_datasets['train'].classes

# TODO: GPU acceleration
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

num_epochs = 25

is_model_ready = False

def reload_model_from_file():
    global is_model_ready
    
    is_model_ready = True
    model_conv.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

def predict(file_path):    
    if not is_model_ready:
        reload_model_from_file()
    
    model = model_conv
    
    was_training = model.training
    model.eval()

    img = Image.open(file_path).convert("RGB")
    img = preprocess['val'](img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img.to(device))
        probabilities = F.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
        return class_names[preds.item()], confidence.item()
    
    model.train(mode=was_training)
    return "prediction failed", 0.0

def main():
    should_train = input('train? y/n: ')

    if should_train == 'y':
        should_multithread = input('use multithreading? y/n: ')
        if should_multithread == 'y':
            for x in ['train', 'val']:
                dataloaders[x] = torch.utils.data.DataLoader(
                    image_datasets[x], batch_size=4, shuffle=True, num_workers=4)

        train()
    else:
        if not is_model_ready:
            reload_model_from_file()
    
    visualize_model(model_conv)
    plt.ioff()
    plt.show()

    input("press enter to quit")
    sys.exit()

def train():
    global is_model_ready
    global model_conv
    
    if is_model_ready:
        # TODO
        # FIXME
        print("please dont call train if you have already done a prediction :)")
    
    model = model_conv
    optimizer = optimizer_conv
    scheduler = exp_lr_scheduler
    
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        temp_model_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), temp_model_path)
        best_acc = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        best_confusion = 0.0
        best_average_precision = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                num_true_pos = 0
                num_true_neg = 0
                num_false_pos = 0
                num_false_neg = 0

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
                    num_true_pos += ((preds == 1) & (target == 1)).sum().item()
                    num_false_pos += ((preds == 1) & (target == 0)).sum().item()
                    num_true_neg += ((preds == 0) & (target == 0)).sum().item()
                    num_false_neg += ((preds == 0) & (target == 1)).sum().item()                    
                    running_loss += loss.item() * inputs.size(0)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = (num_true_pos + num_true_neg) / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_precision = num_true_pos / (num_true_pos + num_false_pos)
                    best_recall = num_true_pos / (num_true_pos + num_false_neg)
                    best_f1 = 2 * ((best_precision * best_recall) / (best_precision + best_recall))
                    print("using this epoch")
                    torch.save(model.state_dict(), temp_model_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best accuracy: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(temp_model_path, weights_only=True))

        torch.save(model.state_dict(), model_path)

        statistics = {
            "accuracy": best_acc,
            "precision": best_precision,
            "recall": best_recall,
            "f1": best_f1,
            "confusion": best_confusion,
            "average_precision": best_average_precision
        }

        print(statistics)

    is_model_ready = True
    model_conv = model
    return statistics

corrections = {}

def retrain_model():
    """retrains images"""
    print("retraining model")
    print(corrections)

def record_correction(file_path, true_label):
    """Keeps track of corrections for later retraining"""
    global corrections
    
    if true_label == 'Normal':
        corrections[file_path] = 0
    elif true_label == 'Pneumonia':
        corrections[file_path] = 1

def visualize_model(model, num_images=6):
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
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

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
