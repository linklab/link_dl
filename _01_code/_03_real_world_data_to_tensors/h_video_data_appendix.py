# pip install imageio[ffmpeg]
import torch
import os
import imageio

video_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "g_video-cockatoo", "cockatoo.mp4")

reader = imageio.get_reader(video_path)
print(type(reader))
meta = reader.get_meta_data()
print(meta)

for i, frame in enumerate(reader):
  frame = torch.from_numpy(frame).float()  # frame.shape: [360, 480, 3]
  print(i, frame.shape)   # i, torch.Size([360, 480, 3])

n_channels = 3
n_frames = 529
video = torch.empty(1, n_frames, n_channels, *meta['size'])  # (1, 529, 3, 480, 360)
print(video.shape)

for i, frame in enumerate(reader):
  frame = torch.from_numpy(frame).float()       # frame.shape: [360, 480, 3]
  frame = torch.permute(frame, dims=(2, 1, 0))  # frame.shape: [3, 480, 360]
  video[0, i] = frame

video = video.permute(dims=(0, 2, 1, 3, 4))
print(video.shape)
