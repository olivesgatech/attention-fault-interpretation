import torch
from torch.utils.data import Dataset
import os
import numpy as np
from scipy import signal


def gen_dist_mask(fault_image, smoothing=0.01, emphasis=10):
    """function generates for a given fault image a mask of similar dimensions weighing each 
    pixel by its proximity to the fault"""
    
    mask = np.zeros(fault_image.shape)
    
    for i in range(fault_image.shape[0]):
        columns, rows = np.meshgrid(np.arange(fault_image[i].shape[0]), np.arange(fault_image[i].shape[1]))
        img_idcs = np.hstack((rows.flatten().reshape(-1,1), columns.flatten().reshape(-1,1))).reshape(1,-1)
    
        fault_idcs = np.tile(np.argwhere(fault_image[i]==1), img_idcs.shape[1]//2)
        
        if fault_idcs.size == 0:
            mask += 1e-3
            mask = torch.from_numpy(mask).to(torch.float32).to('cuda')
            return mask
        
        else:
        
            point_distances = (img_idcs - fault_idcs)**2
            point_distances = signal.convolve2d(point_distances, np.array([[1,1]]), 'valid')[:,::2]
            projections = np.min(point_distances, axis=0)
        
            mask[i][img_idcs.reshape(-1,2)[:,0], img_idcs.reshape(-1,2)[:,1]] = projections
    
    mask = emphasis*np.exp(-smoothing*mask)
    mask = torch.from_numpy(mask).to(torch.float32).to('cuda')
    
    return mask

def _gen_picks(dim1, dim2, sampling_resolution, label_inds):
    """Function generates coordinates for cube samples to be taken from a 3D
    seismic volume.
    
    Parameters
    ----------
    dim1: int
        integer speifying height of each crossline
        
    dim2: int
        integer specifying width of each crossline
        
    sampling_resolution: int
        number of points to be sampled along each dimension
        
    label_inds: list
        integer list specifying sampling indices of labeled crosslines to be
        used for training.
        
    Returns
    -------
    sampling_idcs: array_like
        (N, 3) dimensional array specifying coordinate triplets in the seismic
        volume in the order (height_sample, inline_sample, crossline_sample).
    """
    
    h , w = dim1, dim2
    depth_samples = np.linspace(0, h, sampling_resolution)
    width_samples = np.linspace(0, w, sampling_resolution)
    width_grid, depth_grid = np.meshgrid(width_samples, depth_samples)
    sample_idcs = np.stack((depth_grid.flatten().astype(int), width_grid.flatten().astype(int))).T
    sample_idcs = np.tile(sample_idcs, (len(label_inds),1))
    sample_idcs = np.concatenate((sample_idcs, np.zeros((sample_idcs.shape[0],1))), axis=1)
    for i, label_num in enumerate(label_inds):
        length = depth_grid.size
        sample_idcs[length*i:length*(i+1),2] = label_num
        
    sample_idcs = np.random.permutation(sample_idcs).astype(int)
    
    return sample_idcs

class SeismicForwardMC(Dataset):
    def __init__(self, Picks, SeismicVolume, Win=[64, 64, 64], toTensor=True, decimate=False, transform=None):
        """
        Args:
            Picks (2D Array): Time,IL,XL.
            SeismicVolume (3D Array): Time,XL,IL.
            hWin (List): Time,XL,IL
            decimate (bool): Decimate subvolume to 2X2
            transform (bool): Pass data to Transforms
        """
        self.Picks = Picks
        self.D = SeismicVolume
        self.transform = transform
        self.Win = Win
        self.toTensor = toTensor
        self.decimate = decimate

    def __len__(self):
        return np.size(self.Picks, axis=0)

    def __getitem__(self, idx):

        def Window(D, Pick, ht=64, hxl=64, hil=64, decimate=False):
            # D = 3D seismic Volume
            a, b, c = Pick[0], Pick[1], Pick[2]
            d = D[a:a + ht, b:b + hxl, c:c + hil]
            if decimate:
                return d[::2, ::2, ::2]
            else:
                return d, [a, b, c]

        image, I = Window(self.D, self.Picks[idx],
                          ht=self.Win[0], hxl=self.Win[1], hil=self.Win[2],
                          decimate=self.decimate)

        if self.toTensor:
            image = torch.from_numpy(image[np.newaxis, ...]).float()

        return {'image': image, 'Ind': I}
    
    
class SeismicFinetuneMC(Dataset):
    def __init__(self, seismic, labels, label_inds, win_size=128, sampling_resolution=10):
        self.seismic = seismic
        self.labels = labels
        self.win_size = win_size
        self.picks = _gen_picks(seismic.shape[0], seismic.shape[2], sampling_resolution=sampling_resolution, label_inds=label_inds)
        
    def __len__(self):
        return self.picks.shape[0]
    
    def __getitem__(self, idx):
        triplet = self.picks[idx]  # get section number
        seismic_tensor = np.zeros((3, self.win_size, self.win_size, self.win_size))
        label_tensor =  np.zeros((self.win_size, self.win_size, self.win_size))
        
        h, w, d = triplet[0], triplet[1], triplet[2] # get height, width, and depth coordinates
        cube = self.seismic[h:h+self.win_size, d:d+self.win_size, w:w+self.win_size]
        label = self.labels[h:h+self.win_size, d:d+self.win_size, w:w+self.win_size]
        
        # pad if needed
        pad_h, pad_w, pad_d = self.win_size - cube.shape[0], self.win_size - cube.shape[2], self.win_size - cube.shape[1]
        cube = np.pad(cube, ((0,pad_h),(0,pad_d),(0,pad_w)), 'constant')
        label = np.pad(label, ((0,pad_h),(0,pad_d),(0,pad_w)), 'constant')

        seismic_tensor = torch.from_numpy(cube[np.newaxis,...]).float().cuda()
        label_tensor = torch.from_numpy(label).float().cuda()
        
        return seismic_tensor, label_tensor
            
        
class SeismicIntelligentFinetuneMC(Dataset):
    def __init__(self, seismic, labels, win_size=128, max_points=36):
        self.seismic = seismic
        self.labels = labels
        self.win_size = win_size
        self.picks = np.random.permutation(np.argwhere(self.labels))[:max_points].astype(int)

    def __len__(self):
        return self.picks.shape[0]
    
    def __getitem__(self, idx):
        triplet = self.picks[idx]  # get section number
        h, w, d = triplet[0], triplet[2], triplet[1] # get height, width, and depth coordinates
        h, w = np.random.poisson(h), np.random.poisson(w) 
        
        cube = self.seismic[h-self.win_size//2:h+self.win_size//2, d:d+self.win_size, w-self.win_size//2:w+self.win_size//2]
        label = self.labels[h-self.win_size//2:h+self.win_size//2, d:d+self.win_size, w-self.win_size//2:w+self.win_size//2]
        
        # pad if needed
        pad_h, pad_w, pad_d = self.win_size - cube.shape[0], self.win_size - cube.shape[2], self.win_size - cube.shape[1]
        cube = np.pad(cube, ((0,pad_h),(0,pad_d),(0,pad_w)), 'constant')
        label = np.pad(label, ((0,pad_h),(0,pad_d),(0,pad_w)), 'constant')

        seismic_tensor = torch.from_numpy(cube[np.newaxis,...]).float().cuda()
        label_tensor = torch.from_numpy(label).float().cuda()
        
        return seismic_tensor, label_tensor
    

# dataset class for pretraining on synthetic data
class SyntheticFaultDataset(Dataset):
        def __init__(self, seismic_path, fault_path):
            
            self.filenames = os.listdir(seismic_path)
            
            # list of paths to synthetic seismic cubes
            self.seismic_paths = [os.path.join(seismic_path, file) for file in 
                                  self.filenames if 
                                  file.endswith('.dat')]  
            
            # list of paths to fault cubes
            self.fault_paths = [os.path.join(fault_path, file) for file in 
                               self.filenames]  # label directory
            
        def __len__(self):
            return len(self.seismic_paths)
        
        def __getitem__(self, idx):
            
            # load seismic cube
            seismic = np.fromfile(self.seismic_paths[idx], 
                                  dtype=np.single).reshape(1,128,128,128)
            
            seismic = seismic / seismic.std()  # standardize
            
            # load fault cube
            fault = np.fromfile(self.fault_paths[idx], 
                                dtype=np.single).reshape(128,128,128)
            
            # convert to torch tensors
            seismic_tensor = torch.from_numpy(seismic.transpose(0,3,2,1)).float().cuda()
            fault_tensor = torch.from_numpy(fault.transpose(2,1,0)).float().cuda()
            
            return seismic_tensor, fault_tensor
            
            


