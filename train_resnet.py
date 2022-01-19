# Training file for resnet 

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class FishFinDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fish_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fish_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.fish_frame.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.fish_frame.iloc[idx, 1]
        label2 = self.fish_frame[idx,2]
        convert_strings = {'Has_fin':0, 'No_fin':1, 'Cannot_see':2}
        convert_strings2 = {'No Fungi': 0, 'Slight Fungi':1, 'Severe Fungi':2}

        sample = {'image': image, 'label': np.array([convert_strings[label]]), 'label2': np.array([convert_strings2[label2]])}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        label = label.reshape((-1,))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        

        return {'image': img, 'label': label}



#####################################

#KOLLA IGENOM CUSTOMKLASSEN 


#####################################

def my_collate(batch):
    data = [item['image'] for item in batch]
    target = [item['label'] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

#####################################
epochs = 3
batch_size = 2
rescale = Rescale((256,512))

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')



totensor = ToTensor()
tfs = transforms.Compose([rescale,totensor])
trainset = FishFinDataset('annotations.csv','./fin_images',transform=tfs)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=1)

classes = ('Wild', 'Farmed', 'Unknown')


net = torchvision.models.resnet18(pretrained=False)
net.fc= nn.Linear(512, 3)         #Labeling has fin/no fin/unsure
net.float()
net.to(device)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



for epoch in range(epochs):
    running_loss = 0
    for i,data in enumerate(tqdm(trainloader)):
        image = data['image']
        label = data['label']
        optimizer.zero_grad()
        output = net(image.float())
        loss = criterion(output,label.squeeze())
        loss.backward()
        optimizer.step()
        #print(data)