# Training file for resnet 

from __future__ import print_function, division
import os
from turtle import forward
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
        label2 = self.fish_frame.iloc[idx,2]
        convert_strings = {'Has_fin':0, 'No_fin':1, 'Cannot_see':2}
        convert_strings2 = {'No Fungi': 0, 'Slight Fungi':1, 'Severe Fungi':2}

        sample = {'image': image, 'label': np.array([convert_strings[label]]), 'label2': np.array([convert_strings2[label2]])}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, label2 = sample['image'], sample['label'], sample['label2']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        label = label.reshape((-1,))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label),
                'label2':torch.from_numpy(label2)}


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
        image, label, label2 = sample['image'], sample['label'], sample['label2']

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
        

        return {'image': img, 'label': label, 'label2': label2}


class MultiHeadResNet(torch.nn.Module):
    def __init__(self,net):
        super().__init__()
        self.conv1 = list(net.children())[0]
        self.bn1 = list(net.children())[1]
        self.relu = list(net.children())[2]
        self.maxpool = list(net.children())[3]
        self.layer1 = list(net.children())[4]
        self.layer2 = list(net.children())[5]
        self.layer3 = list(net.children())[6]
        self.layer4 = list(net.children())[7]
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512,3)
        self.fc2 = nn.Linear(512,3)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1,x2



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

train_csv_path = "/home/johan/Documents/Projekt/FiDiMo/Vattenfall-fish-open-data/Vattenfall-fish-open-data/combined_dataset/train.csv"
validation_csv_path = "/home/johan/Documents/Projekt/FiDiMo/Vattenfall-fish-open-data/Vattenfall-fish-open-data/combined_dataset/validation.csv"
test_csv_path = "/home/johan/Documents/Projekt/FiDiMo/Vattenfall-fish-open-data/Vattenfall-fish-open-data/combined_dataset/test.csv"
image_root = "/home/johan/Documents/Projekt/FiDiMo/Vattenfall-fish-open-data/Vattenfall-fish-open-data/combined_dataset/combined_images"

totensor = ToTensor()
tfs = transforms.Compose([rescale,totensor])
trainset = FishFinDataset(train_csv_path, image_root ,transform=tfs)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=1)

validationset = FishFinDataset(validation_csv_path, image_root ,transform=tfs)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size,shuffle=True, num_workers=1)

testset = FishFinDataset(test_csv_path, image_root ,transform=tfs)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=True, num_workers=1)

classes = ('Wild', 'Farmed', 'Unknown')


net = torchvision.models.resnet18(pretrained=False)

myNet = MultiHeadResNet(net)

#net.fc= nn.Linear(512, 3)         #Labeling has fin/no fin/unsure
myNet.float()
myNet.to(device)

#Drop the last adaptive pooling and fc layer
#backbone = torch.nn.Sequential(*(list(net.children())[:-2]))
#head1 = torch.nn.Sequential(*(list(net.children())[-2:]))
#head2 = torch.nn.Sequential(*(list(net.children())[-2:]))
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(myNet.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(myNet.parameters(),lr = 0.01)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs")
n_total_steps = len(trainloader)


for epoch in range(epochs):
    running_loss = 0
    for i,data in enumerate(tqdm(trainloader)):
        image = data['image']
        label = data['label']
        label2 = data['label2']
        optimizer.zero_grad()
        out1,out2 = myNet(image.float())
        
        loss1 = criterion(out1,label.squeeze())
        loss2 = criterion(out2,label2.squeeze())
        loss = loss1+loss2
        loss.backward()
        optimizer.step()
        #print(data)
        if (i+1) % 100 == 0:
            writer.add_scalar('training loss', running_loss / 100, epoch*n_total_steps + i)
            running_loss = 0.0
    
    torch.save(myNet.state_dict(), "./saved_epoch_run2_" + str(epoch)+".pt")
    validation_total = 0
    validation1 = 0
    validation2 = 0
    #Run inference on validation set
    for j,data in enumerate(tqdm(validationloader)):
        image = data['image']
        label = data['label']
        label2 = data['label2']

        out1,out2 = myNet(image.float())
        
        loss1 = criterion(out1,label.squeeze())
        loss2 = criterion(out2,label2.squeeze())
        loss = loss1+loss2

    writer.add_scalar('training loss', running_loss / 1000, epoch)
        
        

