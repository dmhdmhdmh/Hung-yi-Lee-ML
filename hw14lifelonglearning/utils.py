import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.utils.data.sampler as sampler
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import tqdm
from tqdm import trange

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0)


# Rotate MNIST to generate 10 tasks

def _rotate_image(image, angle):
    if angle is None:
        return image

    image = transforms.functional.rotate(image, angle=angle)
    return image


def get_transform(angle=None):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: _rotate_image(x, angle)),
                                    Pad(28)
                                    ])
    return transform


class Pad(object):
    def __init__(self, size, fill=0, padding_mode='constant'):
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # If the H and W of img is not equal to desired size,
        # then pad the channel of img to desired size.
        img_size = img.size()[1]
        assert ((self.size - img_size) % 2 == 0)
        padding = (self.size - img_size) // 2
        padding = (padding, padding, padding, padding)
        return F.pad(img, padding, self.padding_mode, self.fill)


class Data():
    def __init__(self, path, train=True, angle=None):
        transform = get_transform(angle)
        self.dataset = datasets.MNIST(root=os.path.join(path, "MNIST"), transform=transform, train=train, download=True)

class Args:
    task_number = 5
    epochs_per_task = 10
    lr = 1.0e-4
    batch_size = 128
    test_size=8192

args=Args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generate rotations for the tasks.

# generate rotated MNIST data from 10 different rotations.

angle_list = [20 * x for x in range(args.task_number)]

# prepare rotated MNIST datasets.

train_datasets = [Data('data', angle=angle_list[index]) for index in range(args.task_number)]
train_dataloaders = [DataLoader(data.dataset, batch_size=args.batch_size, shuffle=True) for data in train_datasets]

test_datasets = [Data('data', train=False, angle=angle_list[index]) for index in range(args.task_number)]
test_dataloaders = [DataLoader(data.dataset, batch_size=args.test_size, shuffle=True) for data in test_datasets]

# Visualize label 0-9 1 sample MNIST picture in 5 tasks.
sample = [Data('data', angle=angle_list[index]) for index in range(args.task_number)]

plt.figure(figsize=(30, 10))
for task in range(5):
  target_list = []
  cnt = 0
  while (len(target_list) < 10):
    img, target = sample[task].dataset[cnt]
    cnt += 1
    if target in target_list:
      continue
    else:
      target_list.append(target)
    plt.subplot(5, 10, (task)*10 + target + 1)
    curr_img = np.reshape(img, (28, 28))
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'), fignum=False)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.title("task: " + str(task+1) + " " + "label: " + str(target), y=1)

class Model(nn.Module):
  """
  Model architecture
  1*28*28 (input) → 1024 → 512 → 256 → 10
  """
  def __init__(self):
    super(Model, self).__init__()
    self.fc1 = nn.Linear(1*28*28, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 256)
    self.fc4 = nn.Linear(256, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.view(-1, 1*28*28)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.relu(x)
    x = self.fc4(x)
    return x

example = Model()
print(example)


def train(model, optimizer, dataloader, epochs_per_task, lll_object, lll_lambda, test_dataloaders, evaluate, device,
          log_step=1):
    model.train()
    model.zero_grad()
    objective = nn.CrossEntropyLoss()
    acc_per_epoch = []
    loss = 1.0
    bar = tqdm.auto.trange(epochs_per_task, leave=False, desc=f"Epoch 1, Loss: {loss:.7f}")
    for epoch in bar:
        for imgs, labels in tqdm.auto.tqdm(dataloader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = objective(outputs, labels)
            total_loss = loss
            lll_loss = lll_object.penalty(model)
            total_loss += lll_lambda * lll_loss
            lll_object.update(model)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss = total_loss.item()
            bar.set_description_str(desc=f"Epoch {epoch + 1:2}, Loss: {loss:.7f}", refresh=True)
        acc_average = []
        for test_dataloader in test_dataloaders:
            acc_test = evaluate(model, test_dataloader, device)
            acc_average.append(acc_test)
        average = np.mean(np.array(acc_average))
        acc_per_epoch.append(average * 100.0)
        bar.set_description_str(desc=f"Epoch {epoch + 2:2}, Loss: {loss:.7f}", refresh=True)

    return model, optimizer, acc_per_epoch

def evaluate(model, test_dataloader, device):
    model.eval()
    correct_cnt = 0
    total = 0
    for imgs, labels in test_dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, pred_label = torch.max(outputs.data, 1)

        correct_cnt += (pred_label == labels.data).sum().item()
        total += torch.ones_like(labels.data).sum().item()
    return correct_cnt / total

