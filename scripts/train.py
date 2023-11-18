# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:14:11 2022

@author: mustaah
"""

import torch 
import torch.nn as nn
import numpy as np
from core.networks import UNet3D
from core.dataloaders import gen_dist_mask, SeismicIntelligentFinetuneMC
from torch.utils.data import DataLoader
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser(add_help=True, prog='FaultCNN', description='Imports as numpy volume and passes it '
                                                                         'through a Fault Segmentation CNN')
ap.add_argument('-i', '--input', type=str, required=True, 
                help='[str] Path to input seismic volume (crossline x inline x time) in numpy format')
ap.add_argument('-l', '--label', type=str, required=True, 
                help='[str] Path to labels for input seismic volume (crossline x inline x time) in numpy format')
ap.add_argument('-m', '--model', type=str, required=True, 
                help='[str] Path to network weights file (.pt or.pth file)')
ap.add_argument('-o', '--output', type=str, required=True, 
                help='[str] Path to output .pth file containing finetuned model weights')
ap.add_argument('-or', '--ordering', type=int, nargs='+', required=False, 
                default=[0,1,2], help='[list] Ordering of seismic volume')
ap.add_argument('-d', '--decimate', type=bool, required=False, default="False",
                help='[str] finetune on decimated volume')
ap.add_argument('-w', '--window', type=int, required=False, default=128,
                help='[int] Window size')
ap.add_argument('-n', '--num_samples', type=int, required=False, default=100,
                help='[int] Number of cubes to be stochastically sampled in the vicinity of annotated faults each iteration')
ap.add_argument('-e', '--epochs', type=int, required=False, default=20, 
                help='Number of finetuning epochs')
args = vars(ap.parse_args())

num_samples = args['num_samples']  # number of grid points for each dimension
lr=1e-4
epochs = args['epochs']  # finetune epochs
label_inds = args['indices']  # list of labeled sections provided
win_size = args['window']

# import model and load state dict
model = UNet3D().cuda()
model = nn.DataParallel(model)  # for multi-gpu training
model.load_state_dict(torch.load(args['model']))

# data paths
datapath = args['input']
labelpath = args['label']

# load prepreocess data
x0, x1, x2 = args['ordering'][0],args['ordering'][1],args['ordering'][2]
seismic = np.load(datapath).astype(np.float32).transpose(x0,x1,x2)
labels = 1 * np.load(labelpath).astype(int).transpose(x0,x1,x2)

if args['decimate']=="True":
    seismic, labels = seismic[::3,:,::3], labels[::3,:,::3]
else:
    pass

# standardize the seismic volume
seismic_std = seismic.std()
seismic = np.clip(seismic, -3*seismic_std, 3*seismic_std)
seismic = seismic/seismic_std

# create masked array
mask = np.full(labels.shape, False, dtype=bool)
mask[:,label_inds,:] = True
labels[~mask] = 0

# loaders
dataset = SeismicIntelligentFinetuneMC(seismic, labels, win_size=win_size, max_points=num_samples)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# loss function
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100]).float().cuda(), reduction='none')

# finetune loop
for epoch in range(epochs):
    for i, (cube_data, cube_labels) in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()
        pred_labels = model(cube_data).squeeze(1)

        # weight loss tensor pixels by the attention map
        mask = gen_dist_mask(cube_labels[:,:,0,:].detach().cpu().numpy())
        loss = loss_fn(pred_labels, cube_labels)[:,:,0,:] * mask
        loss = loss.sum() / (win_size**2)

        loss.backward()
        optimizer.step()
        print('Epoch: {} | Iter: {} | Train Loss: {:0.4f}'.format(epoch, i, loss.item()))

    if epoch % 10 == 0:  # save model every 10 epochs
        save_path = args['output']
        torch.save(model.state_dict(), save_path)