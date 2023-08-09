import os
import imageio.v2 as imageio
import torch

img_arr = imageio.imread(os.path.join(os.path.pardir, "data", "a_image-dog", "bobby.jpg"))
print(type(img_arr))
print(img_arr.shape)
print(img_arr.dtype)

img = torch.from_numpy(img_arr)
out = img.permute(2, 0, 1)
print(out.shape)

print("#" * 50, 1)

data_dir = os.path.join(os.path.pardir, "data", "b_image-cats")
filenames = [
    name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.png'
]
print(filenames)

from PIL import Image
for i, filename in enumerate(filenames):
    image = Image.open(os.path.join(data_dir, filename))
    image.show()

batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t[:3]
    batch[i] = img_t

print(batch.shape)

print("#" * 50, 2)

batch = batch.float()
batch /= 255.0
print(batch.dtype)
print(batch.shape)

n_channels = batch.shape[1]

for c in range(n_channels):
    mean = torch.mean(batch[:, c])
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c] - mean) / std
