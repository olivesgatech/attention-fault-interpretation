# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 22:53:25 2023

@author: ahmad
"""

import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from core.dataloaders import SyntheticFaultDataset
from core.networks import UNet3D
from core.utils import load_checkpoint, save_checkpoint
import argparse

# arguments
ap = argparse.ArgumentParser(add_help=True, prog='pretrain', description="Script pretrains a 3D CNN on synthetic fault data")
ap.add_argument('-s', '--seismic', type=str, required=True, help='[str] Path to seismic volume files')
ap.add_argument('-f', '--faults', type=str, required=True, help='[str] Path to fault volume files')
args = vars(ap.parse_args())

# global variable 
use_checkpoint = False

# checkpoint path
checkpoint_path = "../checkpoints"

# paths to data
seismic_path = args['seismic']
fault_path = args['faults']

# set up dataloader
trainset = SyntheticFaultDataset(seismic_path, fault_path)
trainloader = DataLoader(trainset, batch_size=2, shuffle=True)

# initialize model 
model = UNet3D().cuda()
nn.DataParallel(model)  # for multi-gpu training

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# define loss function
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([25]).float().cuda())

# load model, optimizer, and last training epoch if loading from saved checkpoitn
if use_checkpoint:
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path+'/checkpoint.pth')
else:
    start_epoch = 0   

# start training
epochs = 500
try:
    for epoch in range(start_epoch, epochs):
        for itr, (x, y) in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            
            y_pred = model(x).squeeze(1) # run model on seismic cube
            loss = criterion(y_pred, y)
            
            # backpropagate loss
            loss.backward()
            optimizer.step()
            
            # print loss every 20 iterations and checkpoint
            if itr%20==0:
                print('Epoch: {} | Training Loss: {:0.4f}'.format(epoch, loss.item()))
        
        if epoch%2==0:
            print('Checkpointing...')
            save_checkpoint(model, optimizer, epoch, checkpoint_path+'/checkpoint.pth')
            print('Checkpointing Completed!')
                
except KeyboardInterrupt:
    print('Code Interrupted. Checkpointing...')
    save_checkpoint(model, optimizer, epoch, checkpoint_path+'/checkpoint.pth')
    print('Checkpointing Completed. Now exiting')    
            
# save checkpoint once training completes
save_checkpoint(model, optimizer, epoch, checkpoint_path+'/checkpoint.pth')
