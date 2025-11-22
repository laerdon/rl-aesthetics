from typing import *
from dataset.heuristics import *

# want to define a loss function

CANVAS_SIZE = (400, 400)

def compute_loss(canvas, dims, predicted, target):
    # want to check both heuristics and MSE from the target output

    """    
    :param predicted: [(x,y)]
    :param target: [(x,y)]
    """
    mse_loss = 0.0
    for (p, t) in zip(predicted, target):
        mse_loss += (p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2
    mse_loss /= len(predicted)

    if not chk_within_bounds(canvas, dims, predicted):
        mse_loss += 1000.0  # large penalty for going out of bounds
    
    mse_loss += quantify_overlap(canvas, dims, predicted)

    return mse_loss