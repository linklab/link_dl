import os

import imageio.v2 as imageio

dir_path = os.path.join(os.path.pardir, "_00_data", "c_volumetric-dicom", "2-LUNG_3.0_B70f-04083")
vol_array = imageio.volread(dir_path, format='DICOM')
print(type(vol_array))
print(vol_array.shape)
print(vol_array.dtype)
print(vol_array[0])

print("#" * 50, 1)


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 10))
for id in range(0, 99):
    fig.add_subplot(10, 10, id + 1)
    plt.imshow(vol_array[id])
plt.show()

import torch
vol = torch.from_numpy(vol_array).float()
vol = torch.unsqueeze(vol, 0)  # channel
vol = torch.unsqueeze(vol, 0)  # data size

print(vol.shape) # >>> torch.Size([1, 1, 99, 512, 512])

print("#" * 50, 2)

mean = torch.mean(vol, dim=-1, keepdim=True)
print(mean.shape)
std = torch.std(vol, dim=-1, keepdim=True)
print(std.shape)
vol = (vol - mean) / std
print(vol.shape)

print(vol[0, 0])
