import matplotlib.pyplot as plt
import numpy as np
import torch
from model import AE
from preprocess import preprocess

model = AE().cuda()
trainX = np.load('trainX.npy')
trainX_preprocessed = preprocess(trainX)
# 畫出原圖
plt.figure(figsize=(10, 4))
indexes = [1, 2, 3, 6, 7, 9]
imgs = trainX[indexes,]
for i, img in enumerate(imgs):
    plt.subplot(2, 6, i + 1, xticks=[], yticks=[])
    plt.imshow(img)

# 畫出 reconstruct 的圖
inp = torch.Tensor(trainX_preprocessed[indexes,]).cuda()
latents, recs = model(inp)
recs = ((recs + 1) / 2).cpu().detach().numpy()
recs = recs.transpose(0, 2, 3, 1)
for i, img in enumerate(recs):
    plt.subplot(2, 6, 6 + i + 1, xticks=[], yticks=[])
    plt.imshow(img)

plt.tight_layout()