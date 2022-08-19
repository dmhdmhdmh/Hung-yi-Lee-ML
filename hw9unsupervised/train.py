import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
from scipy.cluster.vq import vq, kmeans
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from model import (fcn_autoencoder, conv_autoencoder, VAE, loss_vae)


train = np.load('train.npy', allow_pickle=True)
test = np.load('test.npy', allow_pickle=True)
task = 'ae'

if task == 'knn':
    x = train.reshape(len(train), -1)
    y = test.reshape(len(test), -1)
    scores = list()
    for n in range(1, 10):
      kmeans_x = MiniBatchKMeans(n_clusters=n, batch_size=100).fit(x)
      y_cluster = kmeans_x.predict(y)
      y_dist = np.sum(np.square(kmeans_x.cluster_centers_[y_cluster] - y), axis=1)

      y_pred = y_dist
      # score = f1_score(y_label, y_pred, average='micro')
      # score = roc_auc_score(y_label, y_pred, average='micro')
      # scores.append(score)
    # print(np.max(scores), np.argmax(scores))
    # print(scores)
    # print('auc score: {}'.format(np.max(scores)))

if task == 'pca':
    x = train.reshape(len(train), -1)
    y = test.reshape(len(test), -1)
    pca = PCA(n_components=2).fit(x)

    y_projected = pca.transform(y)
    y_reconstructed = pca.inverse_transform(y_projected)
    dist = np.sqrt(np.sum(np.square(y_reconstructed - y).reshape(len(y), -1), axis=1))

    y_pred = dist
    # score = roc_auc_score(y_label, y_pred, average='micro')
    # score = f1_score(y_label, y_pred, average='micro')
    # print('auc score: {}'.format(score))

if task == 'ae':
    num_epochs = 1000
    batch_size = 128
    learning_rate = 1e-3

    # {'fcn', 'cnn', 'vae'}
    model_type = 'cnn'

    x = train
    if model_type == 'fcn' or model_type == 'vae':
        x = x.reshape(len(x), -1)

    data = torch.tensor(x, dtype=torch.float)
    train_dataset = TensorDataset(data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    model_classes = {'fcn': fcn_autoencoder(), 'cnn': conv_autoencoder(), 'vae': VAE()}
    model = model_classes[model_type].cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate)

    best_loss = np.inf
    model.train()
    for epoch in range(num_epochs):
        for data in train_dataloader:
            if model_type == 'cnn':
                img = data[0].transpose(3, 1).cuda()
            else:
                img = data[0].cuda()
            # ===================forward=====================
            output = model(img)
            if model_type == 'vae':
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            else:
                loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================save====================
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model, 'best_model_{}.pt'.format(model_type))
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))
