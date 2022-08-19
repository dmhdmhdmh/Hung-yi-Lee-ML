from skimage import transform,exposure
from sklearn import model_selection, preprocessing, metrics, feature_selection
import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
import time
from model import Classifier
import cv2

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x
#testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

workspace_dir = 'D:/homeworkpy/Data/hw3/food-11/food-11'
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))
batch_size = 128
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

save_path = 'D:/homeworkpy/hw3CNNclassification'
modelfiles = []

for model_name in os.listdir(save_path):
    if model_name.endswith('.pth'):
        model_path_final = os.path.join(save_path,model_name)
        modelfiles.append(model_path_final)

model_best = Classifier.cuda()
for m in modelfiles:
    model_best.load_state_dict(torch.load(m))
    model_best.eval()
    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model_best(data.cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)

    #將結果寫入 csv 檔
    with open(f"predict{m}.csv", 'w') as f:
        f.write('Id,Category\n')
        for i, y in  enumerate(prediction):
            f.write('{},{}\n'.format(i, y))