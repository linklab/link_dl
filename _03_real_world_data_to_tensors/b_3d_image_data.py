import os
import numpy as np

import imageio.v2 as imageio

dir_path = os.path.join(os.path.pardir, "_00_data", "c_volumetric-dicom", "2-LUNG_3.0_B70f-04083")
vol_arr = imageio.volread(dir_path, format='DICOM')
print(type(vol_arr))
print(vol_arr.shape)
print(vol_arr.dtype)
print(vol_arr[0])

print("#" * 50, 1)


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 10))
for id in range(0, 99):
    fig.add_subplot(10, 10, id + 1)
    plt.imshow(vol_arr[id])
plt.show()

import torch
vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol, 0)  # channel
vol = torch.unsqueeze(vol, 0)  # data size

print(vol.shape) # >>> torch.Size([1, 1, 99, 512, 512])

print("#" * 50, 2)

for depth in range(99):
    mean = torch.mean(vol[0, depth, :])
    std = torch.std(vol[0, depth, :])
    vol[0, depth, :] = (vol[0, depth, :] - mean) / std

print(vol[0, 0])
