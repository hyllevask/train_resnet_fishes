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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

#Import the user defined functions from the helper file.
from net_and_functions import FishFinDataset, ToTensor, Rescale, MultiHeadResNet


#####################################

def main(mode):

    results_folder = datetime.datetime.now().strftime("%Y-%m-%d %X") + " Mode:"+mode
    os.mkdir(os.path.join("./results", results_folder))
    epochs = 25
    batch_size = 32
    rescale = Rescale((256,512))
    totensor = ToTensor()
    tfs = transforms.Compose([rescale,totensor])

    #CHeck if we are on the gpu-tower
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    train_csv_path = "./combined_dataset/train.csv"
    validation_csv_path = "./combined_dataset/validation.csv"
    #test_csv_path = "./combined_dataset/test.csv"  Dont need test here!
    image_root = "./combined_dataset/combined_images"

    
    
    trainset = FishFinDataset(train_csv_path, image_root ,transform=tfs)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=1)

    validationset = FishFinDataset(validation_csv_path, image_root ,transform=tfs)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size,shuffle=True, num_workers=1)


    #Dont need test here
    #testset = FishFinDataset(test_csv_path, image_root ,transform=tfs)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=True, num_workers=1)

    classes = ('Wild', 'Farmed', 'Unknown')


    net = torchvision.models.resnet18(pretrained=False)

    if mode == "joint":
        myNet = MultiHeadResNet(net)
    else:
        net.fc= nn.Linear(512, 3) #Labeling has fin/no fin/unsure
        myNet = net 

 
    myNet.float()
    myNet.to(device)

    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(myNet.parameters(),lr = 0.01)

    

    writer = SummaryWriter(os.path.join("./results", results_folder))
    n_total_steps = len(trainloader)

    #Main training loop
    for epoch in range(epochs):
        running_loss = 0
        for i,data in enumerate(tqdm(trainloader)):
            
            image = data['image']
            label = data['label']
            label2 = data['label2']
            optimizer.zero_grad()

            if mode == "joint":
                out1,out2 = myNet(image.float().to(device))
            
                loss1 = criterion(out1,label.squeeze().to(device))
                loss2 = criterion(out2,label2.squeeze().to(device))
                loss = loss1+loss2
            elif mode == "fin":
                out = myNet(image.float().to(device))
                loss = criterion(out,label.squeeze().to(device))
            elif mode == "fungi":
                out = myNet(image.float().to(device))
                loss = criterion(out,label2.squeeze().to(device))

            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            #print(data)
            if (i+1) % 100 == 0:
                writer.add_scalar('training loss', running_loss / 100, epoch*n_total_steps + i)
                running_loss = 0.0
                
        torch.save(myNet.state_dict(), "./results/" + results_folder + "/epoch_" + str(epoch)+".pt")
        validation_total = 0
        validation1 = 0
        validation2 = 0

        correct2 = 0
        correct1 = 0
        total1 = 0
        total2 = 0
        total = 0
        correct = 0
        #Run inference on validation set
        for j,data in enumerate(tqdm(validationloader)):
            image = data['image']
            label = data['label']
            label2 = data['label2']
            if mode == "joint":

                out1,out2 = myNet(image.float().to(device))
            
                loss1 = criterion(out1,label.squeeze().to(device))
                loss2 = criterion(out2,label2.squeeze().to(device))
                loss = loss1+loss2
                validation1 += loss1.item()
                validation2 += loss2.item()

                total1 +=  len(label)
                correct1 += (torch.argmax(out1,axis=1) == label.to(device).squeeze()).sum().item() 
                correct2 += (torch.argmax(out2,axis=1) == label2.to(device).squeeze()).sum().item()
                total2 += len(label2)
            elif mode == "fin":
                out = myNet(image.float().to(device))
                loss = criterion(out,label.squeeze().to(device))
                total +=  len(label)
                correct += (torch.argmax(out,axis=1) == label.to(device).squeeze()).sum().item()
            elif mode == "fungi":
                out = myNet(image.float().to(device))
                loss = criterion(out,label2.squeeze().to(device))

                total +=  len(label)
                correct += (torch.argmax(out,axis=1) == label2.to(device).squeeze()).sum().item()
            
            validation_total += loss.item()
            
            


        writer.add_scalar('Validation loss/Total', validation_total / 1000, epoch)
        if mode == "joint":
            writer.add_scalar('Validation loss/Label1', validation1 / 1000, epoch)
            writer.add_scalar('Validation loss/Label2', validation2 / 1000, epoch)
            writer.add_scalar('Validation Accuracy/Label1',correct1/total1, epoch)
            writer.add_scalar('Validation Accuracy/Label2',correct2/total2, epoch)
        else:
            writer.add_scalar('Validation Loss', validation_total / 1000, epoch)
            writer.add_scalar('Validation Accuracy', correct/total, epoch)











if __name__ == "__main__":
    import argparse
   # parser = argparse.ArgumentParser(description="Train script for the first fish dataset")
    #parser.add_argument('--mode',type=str,required=True)
    
   # args = parser.parse_args()
    #mode = args.mode
    mode = "fin"
    allowed_modes = ["fin", "fungi","joint"]

    if mode in allowed_modes:
        main(mode)
    else:
        print("Mode not allowed! Please try again.")
    
        

