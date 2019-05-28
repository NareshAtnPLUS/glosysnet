import numpy as np

def mean_square_error(predicted, target, rms=False):
    mean_square_error = (np.square(predicted - target)).mean()
    rms_mse = np.sqrt(np.square(predicted - target)).mean()
    
    if rms == True:
        return mean_square_error, rms_mse 
    else:
        return mean_square_error
def logistic_loss(y,h_x):
	return np.mean(-y*np.log(h_x) - (1 - y)*np.log(1 - h_x))

def mean_absolute_error(predicted,target):
    return np.absolute(predicted, target).mean()

def mean_bias_error(predicted, target):
    return np.mean(predicted - target)
