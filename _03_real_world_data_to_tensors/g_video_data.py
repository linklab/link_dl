# pip install imageio[ffmpeg]
import torch
import os
import imageio

video_path = os.path.join(os.path.pardir, "data", "g_video-cockatoo", "cockatoo.mp4")

reader = imageio.get_reader(video_path)
print(type(reader))
meta = reader.get_meta_data()
print(meta)

n_channels = 3
n_frames = 529
video = torch.empty(n_channels, n_frames, *meta['size'])
print(video.shape)

for i, frame in enumerate(reader):
    frame = torch.from_numpy(frame).float()     # frame.shape: [360, 480, 3]
    video[:, i] = torch.transpose(frame, 0, 2)

video = video.unsqueeze(dim=0)
print(video.shape)