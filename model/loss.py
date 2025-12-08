from typing import *
from dataset.heuristics import *

# want to define a loss function

# model/loss.py
from typing import *
import torch
from dataset.heuristics import *

def compute_loss(canvas, dims, predicted, target):
    """
    dims: [batch, num_dims] tensor
    predicted: [batch, num_coords] tensor  
    target: [batch, num_coords] tensor
    """
    # mse loss works on batched tensors
    mse_loss = torch.nn.functional.mse_loss(predicted, target)
    
    # heuristic penalties need to loop over batch
    batch_size = dims.shape[0]
    penalty = 0.0
    
    for b in range(batch_size):
        # extract single sample
        dims_sample = dims[b, 2:]
        pred_sample = predicted[b] # predicted doesn't have the canvas dimensions
        
        # check bounds for this sample
        if not chk_within_bounds(canvas, dims_sample, pred_sample):
            penalty += 50.0
        
        # add overlap penalty
        penalty += quantify_overlap(canvas, dims_sample, pred_sample)
    
    # average penalty over batch
    penalty /= batch_size
    
    return mse_loss + penalty