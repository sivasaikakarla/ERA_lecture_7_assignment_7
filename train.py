import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from prettytable import PrettyTable, MARKDOWN
from torchsummary import summary
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CIFAR10
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# Import the model
from model import CIFAR10Net

# 1. Load the CIFAR-10 training dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
data = train_dataset.data
data = data / 255.0
mean = np.mean(data, axis=(0, 1, 2))
std = np.std(data, axis=(0, 1, 2))

# Print the results
print(f"Calculated Mean: {mean}")
print(f"Calculated Std Dev: {std}")

mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.1),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=mean, mask_fill_value=None, p=0.1),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

class AlbumentationDataset(CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

train_dataset = AlbumentationDataset(root='./data', train=True, download=True, transform=train_transforms)
test_dataset = AlbumentationDataset(root='./data', train=False, download=True, transform=test_transforms)

SEED = 2

# CUDA?
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()
print("GPU Available?", device)

# For reproducibility
torch.manual_seed(SEED)

if device == "cuda":
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=128)

# train dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

# Pretty table for collecting all the accuracy and loss parameters in a table
log_table = PrettyTable()

# Display model summary
model_summary = CIFAR10Net().to(device)
summary(model_summary, input_size=(3, 32, 32))


train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y_pred, target)
    #loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

# Main training loop
print("model running on: ", device)
log_table = PrettyTable()
log_table.field_names = ["Epoch", "Training Accuracy", "Test Accuracy", "Diff", "Training Loss", "Test Loss"]

model =  CIFAR10Net().to(device)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.003)
scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

EPOCHS = 50
for epoch in range(EPOCHS):
    print("EPOCH:", epoch+1)
    train(model, device, train_loader, optimizer, epoch)
    #scheduler.step()
    test(model, device, test_loader)
    log_table.add_row([epoch+1, f"{train_acc[-1]:.2f}%", f"{test_acc[-1]:.2f}%", f"{float(train_acc[-1]) - float(test_acc[-1]):.2f}" ,f"{train_losses[-1]:.4f}", f"{test_losses[-1]:.4f}"])
log_table.set_style(MARKDOWN)
print(log_table)