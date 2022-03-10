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
from net_and_functions import FishFinDataset, ToNormTensor, Rescale, MultiHeadResNet


#####################################


# Function writes the hyperparameters to a txt file in the results folder.
def write_hyper_parameters(folderpath,h_param):
    import json
    filename = folderpath + '/hyperparameters.txt'
    with open(filename,'w') as write_dict:
        write_dict.write(json.dumps(h_param))

def main(mode,hyper_param):


    #Get current datatime and create a resultsfolder
    results_folder = datetime.datetime.now().strftime("%Y-%m-%d %X") + " Mode:"+mode
    os.mkdir(os.path.join("./results", results_folder))
    #Write the hyperparameters to a log-file
    write_hyper_parameters(os.path.join("./results", results_folder),hyper_param)


    
    batch_size = 8
    rescale = Rescale((256,512))
    tonormtensor = ToNormTensor()
    tfs = transforms.Compose([rescale,tonormtensor])

    #Check if we are on the gpu-tower
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    train_csv_path = "./combined_dataset/Cleaned_Data/NoZeros/train_clean.csv"
    validation_csv_path = "./combined_dataset/Cleaned_Data/NoZeros/validation_clean.csv"
    #test_csv_path = "./combined_dataset/test.csv"  Dont need test here!
    image_root = "./combined_dataset/combined_images"

    
    
    trainset = FishFinDataset(train_csv_path, image_root ,transform=tfs)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=1)

    validationset = FishFinDataset(validation_csv_path, image_root ,transform=tfs)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size,shuffle=True, num_workers=1)





    net = torchvision.models.resnet18(pretrained=False)

    if mode == "joint":
        myNet = MultiHeadResNet(net)
    else:
        net.fc= nn.Linear(512, 1) #Labeling has fin/no fin/unsure  ResNet18
        #net.fc= nn.Linear(2048, 2) #Labeling has fin/no fin/unsure  RwsNet50
        myNet = net 

 
    myNet.float()
    myNet.to(device)
    #myNet.load_state_dict(torch.load("./results/2022-03-08 12:20:32 Mode:fungi/epoch_50.pt"))
    

    #criterion = nn.CrossEntropyLoss()  #For MultiClass
    criterion = nn.BCEWithLogitsLoss()  #For binary
    optimizer = optim.Adam(myNet.parameters(),lr = 0.02)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    

    writer = SummaryWriter(os.path.join("./results", results_folder))
    n_total_steps = len(trainloader)

    #Main training loop
    for epoch in range(hyper_param['epochs']):
        running_loss = 0
        for i,data in enumerate(tqdm(trainloader)):
            
            image = data['image']
            label = data['label']
            label2 = data['label2']
            optimizer.zero_grad()

            if mode == "joint":
                out1,out2 = myNet(image.float().to(device))
            
                loss1 = criterion(out1,label.to(device))
                loss2 = criterion(out2,label2.to(device))
                loss = loss1+loss2
            elif mode == "fin":
                out = myNet(image.float().to(device))
                loss = criterion(out,label.to(device))
            elif mode == "fungi":
                out = myNet(image.float().to(device))
                loss = criterion(out[:,-1],label2.float().to(device))

            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            #print(data)
            if (i+1) % 50 == 0:
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
            
                loss1 = criterion(out1,label.to(device))
                loss2 = criterion(out2,label2.to(device))
                loss = loss1+loss2
                validation1 += loss1.item()
                validation2 += loss2.item()

                total1 +=  len(label)
                correct1 += (torch.argmax(out1,axis=1) == label.to(device)).sum().item() 
                correct2 += (torch.argmax(out2,axis=1) == label2.to(device)).sum().item()
                total2 += len(label2)
            elif mode == "fin":
                out = myNet(image.float().to(device))
                loss = criterion(out,label.to(device))
                total +=  len(label)
                correct += (torch.argmax(out,axis=1) == label.to(device)).sum().item()
            elif mode == "fungi":
                out = myNet(image.float().to(device))
                loss = criterion(out[:,-1],label2.float().to(device))

                total +=  len(label)
                correct += (torch.round(torch.sigmoid(out[:,-1])) == label2.to(device)).sum().item()
            
            validation_total += loss.item()
            
        
        if mode == "joint":
            writer.add_scalar('Validation Loss/Total', validation_total / len(validationset), epoch)
            writer.add_scalar('Validation Loss/Label1', validation1 / len(validationset), epoch)
            writer.add_scalar('Validation Loss/Label2', validation2 / len(validationset), epoch)
            writer.add_scalar('Validation Accuracy/Label1',correct1/total1, epoch)
            writer.add_scalar('Validation Accuracy/Label2',correct2/total2, epoch)
        else:
            writer.add_scalar('Validation Loss', validation_total / len(validationset)*batch_size, epoch)   #Loss per batch
            writer.add_scalar('Validation Accuracy', correct/total, epoch)
        scheduler.step()











if __name__ == "__main__":
    debug = 1
    if debug == 1:
        mode ='fungi'
        h_param = {
            'epochs': 100,
            'lr': 0.01,
            'load': None,
            'batch': 8
            }
    else:

        import argparse
        parser = argparse.ArgumentParser(description="Train script for the first fish dataset")
        parser.add_argument('--mode',type=str,required=True)
        parser.add_argument('--lr',type=float,default=0.01)
        parser.add_argument('--epochs',type=int,default=30)
        parser.add_argument('--load',type=str,default=None)
        parser.add_argument('--batch',type=int,default=8)

        args = parser.parse_args()
        mode = args.mode

        h_param = {
            'lr': args.lr,
            'epochs': args.epochs,
            'load': args.load,
            'batch': args.batch
        }

    allowed_modes = ["fin", "fungi","joint"]

    if mode in allowed_modes:
        main(mode,h_param)
    else:
        print("Mode not allowed! Please try again.")
    





