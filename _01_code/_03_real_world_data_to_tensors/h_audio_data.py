import torch
import os
import scipy.io.wavfile as wavfile

audio_1_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "f_audio-chirp", "1-100038-A-14.wav")
audio_2_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "f_audio-chirp", "1-100210-A-36.wav")

freq_1, waveform_arr_1 = wavfile.read(audio_1_path)
print(freq_1)
print(type(waveform_arr_1))
print(len(waveform_arr_1))
print(waveform_arr_1)

freq_2, waveform_arr_2 = wavfile.read(audio_2_path)

waveform = torch.empty(2, 1, 220_500)
waveform[0, 0] = torch.from_numpy(waveform_arr_1).float()
waveform[1, 0] = torch.from_numpy(waveform_arr_2).float()
print(waveform.shape)

print("#" * 50, 1)

from scipy import signal

_, _, sp_arr_1 = signal.spectrogram(waveform_arr_1, freq_1)
_, _, sp_arr_2 = signal.spectrogram(waveform_arr_2, freq_2)

sp_1 = torch.from_numpy(sp_arr_1)
sp_2 = torch.from_numpy(sp_arr_2)
print(sp_1.shape)
print(sp_2.shape)

sp_left_t = torch.from_numpy(sp_arr_1)
sp_right_t = torch.from_numpy(sp_arr_2)
print(sp_left_t.shape)
print(sp_right_t.shape)

sp_t = torch.stack((sp_left_t, sp_right_t), dim=0).unsqueeze(dim=0)
print(sp_t.shape)
