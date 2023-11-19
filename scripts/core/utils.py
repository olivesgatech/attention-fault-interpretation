# script contains functional utilities used for various purposes throughout 
# the project

import torch


def save_checkpoint(model, optimizer, epoch, filename):
    # Create a checkpoint dictionary
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'last_epoch':epoch
    }
            
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    # Load the checkpoint dictionary from a file
    checkpoint = torch.load(filename)

    # Load the model and optimizer parameters from the checkpoint
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['last_epoch']
    
    return model, optimizer, start_epoch