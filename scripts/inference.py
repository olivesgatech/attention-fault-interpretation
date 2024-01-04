# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:15:28 2022

@author: mustaah
"""

import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from core.networks import UNet3D
from core.dataloaders import SeismicForwardMC
import argparse
import os
from os.path import join


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser(add_help=True, prog='FaultCNN',
    description='Imports as numpy volume and passes it through a Fault Segmentation CNN')
ap.add_argument('-i', '--input', type=str, required=True, 
                help='[str] Path to input seismic volume of shape (d1 x d2 x d3) in .npy format')
ap.add_argument('-or', '--ordering', type=int, nargs='+', required=False, 
                default=[0,1,2], help='[list] Ordering of dimensions of the volume. Note that the time dimension must be ordered to be the first axis')
ap.add_argument('-m', '--model', type=str, required=True, 
                help='[str] Path to network weights file (.pt or.pth file)')
ap.add_argument('-o', '--output', type=str, required=True, 
                help='[str] Path to directory to store output fault volume file (and images, if specified below)')
ap.add_argument('-d', '--decimate', type=str, required=False, default="False",
                help='[str] Decimate volume by a factor of 3 before inference if True. Default is False.')
ap.add_argument('-w', '--window', type=int, required=False, default=128,
                help='[int] Window size for sample cubes in the volume')
ap.add_argument('-p', '--print', type=str, required=False, default="True",
                help='[str] Print select images from volume overlaid with fault predictions if True')
args = vars(ap.parse_args())


# import model and load state dict
model = UNet3D().cuda()
model = nn.DataParallel(model)  # for multi-gpu inference
model.load_state_dict(torch.load(args['model'])['model'])


# load preprocess data
seismic = np.load(args['input']).astype(np.float32)
x0, x1, x2 = args['ordering'][0],args['ordering'][1],args['ordering'][2]
seismic = seismic.transpose(x0, x1, x2)

if args['decimate']=="True":
    seismic = seismic[::3,:,::3]
else:
    pass

seismic = np.clip(seismic, -3*seismic.std(), 3*seismic.std())
seismic = (seismic - seismic.mean())/seismic.std()

# inference
sample_spacing = args['window']
window_size = args['window']

cube_dims = seismic.shape  # get volume dimensions

# set up grid of sample indices
dim1, dim2, dim3 = np.meshgrid(np.arange(0, cube_dims[0], sample_spacing), np.arange(0, cube_dims[1], sample_spacing),
                               np.arange(0, cube_dims[2], sample_spacing))

# index triplets
index_triplets = np.stack((dim1.flatten(), dim2.flatten(), dim3.flatten()), axis=-1)

# create dataset
seismic_dataset = SeismicForwardMC(index_triplets, seismic, Win=[window_size, window_size, window_size])
fault_volume = np.zeros(seismic.shape)  # volume to store fault probs

# inference and stitching loop
model.eval()
with torch.no_grad():
    for i in range(seismic_dataset.__len__()):
        image = seismic_dataset[i]['image'].unsqueeze(0).cuda()
        
        # pad image to desired size of needed
        if (image.shape[2],image.shape[3],image.shape[4]) != (window_size, window_size, window_size):
            # calculate padding sizes
            pad1 = window_size - image.shape[2]
            pad2 = window_size - image.shape[3]
            pad3 = window_size - image.shape[4]
            
            image = torch.nn.functional.pad(image, (0,pad3,0,pad2,0,pad1))
        
        fault_pred = torch.sigmoid(model(image)).detach().cpu().numpy().squeeze()
        inds = seismic_dataset[i]['Ind']
        a0, a1, a2 = seismic_dataset[i]['image'].shape[1], seismic_dataset[i]['image'].shape[2], seismic_dataset[i]['image'].shape[3],
        fault_volume[inds[0]:inds[0]+a0, inds[1]:inds[1]+a1, inds[2]:inds[2]+a2] = fault_pred[:a0,:a1,:a2]
        
fault_directory = args['output']
if os.path.exists(fault_directory)==False:
    os.mkdir(fault_directory)
else:
    pass
np.save(fault_directory+'/fault_volume.npy', fault_volume.transpose(x0,x1,x2).transpose(x0,x1,x2).astype(np.float32))        

# plot a few sections
if args['print']=="True":
    img_directory = join(args['output'], 'images')
    
    if os.path.exists(img_directory)==False:
        os.mkdir(img_directory)
    else:
        pass
    
    for i in np.linspace(0, seismic.shape[1]-1, 10).astype(int):
        plt.figure(figsize=(12,10))
        plt.imshow(seismic[:,i,:], cmap='seismic')
        plt.imshow(fault_volume[:,i,:], cmap="Greys", alpha=0.4)
        plt.title('Section {}'.format(i))
        plt.axis('off')
        plt.savefig(img_directory+'/section_'+str(i)+'.png')


