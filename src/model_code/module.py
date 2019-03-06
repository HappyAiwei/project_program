import numpy as np

def RMSE(x, y):
    rmse = np.sqrt(np.mean((x.values -y.reshape(-1,1))**2))
    return rmse