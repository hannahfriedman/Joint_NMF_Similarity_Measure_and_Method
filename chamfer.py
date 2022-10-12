import torch
from chamferdist import ChamferDistance
from scipy.io import loadmat
import numpy as np


data = loadmat('swimmer.mat')['X']
data = np.reshape(data, (1, *data.shape))
R = np.random.uniform(size=data.shape)


source_cloud = torch.tensor(data).float()
target_cloud = torch.tensor(R).float()
print(type(target_cloud))
print(target_cloud.shape)
chamferDist = ChamferDistance()
dist_forward = chamferDist(source_cloud, target_cloud, bidirectional = True)
print(dist_forward.detach().cpu().item())
