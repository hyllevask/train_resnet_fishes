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
from net_and_functions import FishFinDataset


#####################################


# Function writes the hyperparameters to a txt file in the results folder.
def write_hyper_parameters(folderpath,h_param,tfs):
    import json
    filename = folderpath + '/hyperparameters.txt'
    with open(filename,'w') as write_dict:
        write_dict.write(json.dumps(h_param))
        write_dict.write(json.dumps(str(tfs)))



def main(mode,hyper_param):


    #Get current datatime and create a resultsfolder
    results_folder = datetime.datetime.now().strftime("%Y-%m-%d %X") + " Mode:"+mode
    os.mkdir(os.path.join("./results", results_folder))
    


    
    batch_size = hyper_param['batch']

    if hyper_param['augment']:
        tfs = transforms.Compose([transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),transforms.RandomRotation(30,expand=True),transforms.RandomHorizontalFlip(p=0.5),transforms.Resize((256,512))])
    else: 
        tfs = transforms.Compose([transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)), transforms.Resize((256,512))])

    #Check if we are on the gpu-tower
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    train_csv_path = "./combined_dataset/Cleaned_Data/NoZeros/train_clean.csv"
    validation_csv_path = "./combined_dataset/Cleaned_Data/NoZeros/validation_clean.csv"
    #test_csv_path = "./combined_dataset/test.csv"  Dont need test here!
    image_root = "./combined_dataset/combined_images"

    
    #Write the hyperparameters to a log-file
    write_hyper_parameters(os.path.join("./results", results_folder),hyper_param,tfs)


    
    trainset = FishFinDataset(train_csv_path, image_root ,transform=tfs)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=1)

    validationset = FishFinDataset(validation_csv_path, image_root ,transform=tfs)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size,shuffle=True, num_workers=1)





    myNet = torchvision.models.resnet18(pretrained=False)

  
    myNet.fc= nn.Linear(512, 1) #ResNet18
    #net.fc= nn.Linear(2048, 2) #RwsNet50
     

 
    myNet.float()
    myNet.to(device)
    #myNet.load_state_dict(torch.load("./results/2022-03-08 12:20:32 Mode:fungi/epoch_50.pt"))
    

    #criterion = nn.CrossEntropyLoss()  #For MultiClass
    criterion = nn.BCEWithLogitsLoss()  #For binary
    optimizer = optim.Adam(myNet.parameters(),lr = hyper_param['lr'])
    
    #Setup tensorboard 
    writer = SummaryWriter(os.path.join("./results", results_folder))
    n_total_steps = len(trainloader)

    #Main training loop
    myNet.train()
    for epoch in range(hyper_param['epochs']):
        running_loss = 0
        for i,data in enumerate(tqdm(trainloader)):
            
            image = data['image'].float().to(device)
            if mode == "fin":
                target = data['label'].float().to(device)
            elif mode == "fungi":
                target = data['label2'].float().to(device)
            optimizer.zero_grad()


            out = myNet(image)
            loss = criterion(out,target)

            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            if (i+1) % 50 == 0:
                writer.add_scalar('training loss', running_loss / 50, epoch*n_total_steps + i)
                running_loss = 0.0
        #Save this epochs last weights
        torch.save(myNet.state_dict(), "./results/" + results_folder + "/epoch_" + str(epoch)+".pt")


        ##################################
        ##          VALIDATION          ##
        ##################################

        #Total Validation Loss
        validation_total = 0

        #Number of total validation samples
        total = 0

        #Number of corect classified samples 
        correct = 0


        #Run inference on validation set
        myNet.eval()
        for j,data in enumerate(tqdm(validationloader)):
            image = data['image'].float().to(device)
            if mode == "fin":
                target = data['label'].float().to(device)
            elif mode == "fungi":
                target = data['label2'].float().to(device)

            out = myNet(image)
            loss = criterion(out,target)

            total +=  len(target)
            correct += (torch.round(torch.sigmoid(out)) == target.to(device)).sum().item()
            
            validation_total += loss.item()
            
        

        writer.add_scalar('Validation Loss', validation_total / len(validationset)*batch_size, epoch)   #Loss per batch
        writer.add_scalar('Validation Accuracy', correct/total, epoch)
    











if __name__ == "__main__":
    debug = 0
    if debug == 1:
        mode ='fungi'
        h_param = {
            'epochs': 100,
            'lr': 0.01,
            'load': None,
            'batch': 8,
            'augment': True
            }
    else:
        print("Using parsed data")
        import argparse
        parser = argparse.ArgumentParser(description="Train script for the first fish dataset")
        parser.add_argument('--mode',type=str,required=True)
        parser.add_argument('--lr',type=float,default=0.01)
        parser.add_argument('--epochs',type=int,default=30)
        parser.add_argument('--load',type=str,default=None)
        parser.add_argument('--batch',type=int,default=8)
        parser.add_argument('--augment',type=bool,default=False)

        args = parser.parse_args()
        mode = args.mode

        h_param = {
            'lr': args.lr,
            'epochs': args.epochs,
            'load': args.load,
            'batch': args.batch,
            'augment': args.augment
        }

    allowed_modes = ["fin", "fungi","joint"]

    if mode in allowed_modes:
        main(mode,h_param)
    else:
        print("Mode not allowed! Please try again.")
    





