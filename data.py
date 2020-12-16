import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import FrequencyMasking, TimeMasking, MelSpectrogram
import os
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torchaudio


class CommandsDataset(Dataset):

    def __init__(self, root_dir, wavs, labels, words, transform=None):
        self.root_dir = root_dir
        self.wav s =wavs
        self.label s =labels
        self.word s =words
        self.transform = transform


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        wav_file = os.path.join(self.root_dir,
                                self.wavs[idx])

        wav = torchaudio.load(wav_file)[0][0]
        lbl = self.labels[idx]
        word = self.words[idx]


        if self.transform:
            wav = self.transform(wav)

        sample = [wav, lbl, word]
        return sample


class LogMelSpectrogram(nn.Module):

    def __init__(self, sample_rate: int = 16000, n_mels: int = 40, masking=True):
        super(LogMelSpectrogram, self).__init__()
        self.transform = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=256, f_min=0, f_max=8000)
        self.masking=masking
        if masking:
          self.freq_masking = FrequencyMasking(freq_mask_param=10)
          self.time_masking = TimeMasking(time_mask_param=30)


    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = self.transform(waveform).squeeze()
        if self.masking:
          spectrogram = self.freq_masking(spectrogram)
          spectrogram = self.time_masking(spectrogram)

        return torch.log(spectrogram + 1e-9)


class Noiser(object):
    def __init__(self, noises_list, alpha_low=0.0001, alpha_high=0.0003, p=1.0):
        self.a_low = alpha_low
        self.a_high=alpha_high
        self.noises_list = noises_list

    def __call__(self, wav):
        noise_name = self.noises_list[np.random.randint(low=0, high=len(self.noises_list))]
        noise = torchaudio.load(noise_name)[0][0]
        l = np.random.randint(low=0, high=len(noise)-len(wav)-1)
        r = l+len(wav)
        noise = noise[l:r]
        alpha = np.random.uniform(low=0.0001, high=0.0003)
        wav = wav + wav*alpha
        wav.clamp_(-1, 1)
        return wav

def my_collate(data):
    wav_batch=[]
    lbls = []
    for sample in data:
      wav_batch.append(sample[0])
      lbls.append(sample[1])
    wav_batch=pad_sequence(wav_batch, batch_first=True)
    lbls=torch.tensor(lbls)
    return wav_batch, lbls

def my_sampler(target):
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler