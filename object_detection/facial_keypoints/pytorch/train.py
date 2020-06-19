import cv2
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from models import Net
from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor

data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])
transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                             root_dir='data/training/',
                                             transform=data_transform)
test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                             root_dir='data/test/',
                                             transform=data_transform)

batch_size = 32

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=2)
test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=2)

net = Net()

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

if  torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    net.cuda()
    print('Training on GPU ...')
else:
    dtype = torch.FloatTensor
    print('Training on CPU ...')


def train_net(n_epochs):
    net.train()
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        test_loss = 0.0
        for batch_i, data in enumerate(train_loader):
            images = data['image']
            key_pts = data['keypoints']

            key_pts = key_pts.view(key_pts.size(0), -1)
            key_pts = key_pts.type(dtype)
            images = images.type(dtype)
            
            output_pts = net(images)

            loss = criterion(output_pts, key_pts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        with torch.no_grad():
            for data in test_loader:
                images = data["image"]
                key_pts = data["keypoints"]
                key_pts = key_pts.view(key_pts.size(0), -1)
                key_pts = key_pts.type(dtype)
                images = images.type(dtype)
                output_pts = net(images)
                loss = criterion(output_pts, key_pts)
                test_loss += loss.item()
            print('Epoch: {}, trainloss: {}, testloss {}'.format(
                epoch + 1,  
                train_loss/len(train_loader), 
                test_loss/len(test_loader)))

    print('Finished Training')

train_net(42)

torch.save(net.state_dict(), "v2_model.pt")