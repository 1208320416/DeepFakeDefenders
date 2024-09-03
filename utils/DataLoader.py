import pandas as pd
import numpy as np
import os, random, torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import torchaudio

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        input = F.conv1d(input, self.flipped_filter)
        return input.squeeze(1)
        
class dataset(object):
    def __init__(self, label_list_path, sample_stride = 1, sample_n_frames = 90, train = False, n_fft = 1024, win_length = 1024, hop_length = 147, a_i = 10, **kwargs):
        label_list = pd.read_csv(label_list_path)
        dirname = os.path.dirname(label_list_path)
        filename = os.path.basename(label_list_path)
        tv = filename.split("_")[0]
        self.video_dir = dirname + '/' + tv
        self.video_names = label_list['video_name'].tolist()
        self.labels = label_list['target'].tolist()

        self.image_stride = sample_stride
        self.num_images = sample_n_frames
        self.num_audio = sample_n_frames * sample_stride * 44100 // 30
        self.a_i = a_i
        self.num_data = self.__len__()
        self.train = train
        
        self.image_preprocessor = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # self.image_preprocessor2 = transforms.Compose([
        #                 transforms.RandomHorizontalFlip(p = 0.5),
        #                 transforms.RandomVerticalFlip(p = 0.5),
        #                 transforms.RandomApply([transforms.RandomRotation(degrees=(-45, 45))], p = 0.5)
        # ])
        
        self.audio_preprocessor = nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.Spectrogram(n_fft = n_fft, win_length = win_length, hop_length=hop_length, window_fn=torch.hamming_window, power=2),
            )

    def __getitem__(self, index):
        video_dir = os.path.join(self.video_dir, self.video_names[index])
        video, audio, info = torchvision.io.read_video(video_dir, pts_unit="sec")

        video_length = len(video)
        clip_length = min(video_length, (self.num_images - 1) * self.image_stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        images = video[start_idx : start_idx + clip_length - 1 : self.image_stride].permute(0,3,1,2).contiguous()  
        images = images / 255.

        if images.shape[0] < self.num_images:
            images = torch.nn.functional.pad(images, (0, 0, 0, 0, 0, 0, 0, self.num_images - len(images)))

        rand_ind = np.random.randint(self.num_data)
        if self.train and self.labels[rand_ind] == 0:
            splice_start  = random.randint(0, self.num_images - 1)
            splice_length = random.randint(0, min(2 * self.num_images // 3, self.num_images - splice_start))
            splice_video, splice_audio, _ = torchvision.io.read_video(os.path.join(self.video_dir, self.video_names[rand_ind]), pts_unit="sec")
            splice_images = splice_video[0 : len(splice_video) : self.image_stride].permute(0,3,1,2).contiguous()  
            splice_images = splice_images / 255.
            del splice_video
            if splice_images.shape[0] < splice_length:
                splice_images = torch.nn.functional.pad(splice_images, (0, 0, 0, 0, 0, 0, 0, splice_length - len(splice_images)))
            splice_images =  transforms.Resize(images.shape[-2:])(splice_images)
            images[splice_start : splice_start + splice_length] = splice_images[ : splice_length]

        images = self.image_preprocessor(images)   # (N,C,H,W)

        start_idx = start_idx * 44100 // 30
        audio = audio[..., start_idx : min(start_idx + self.num_audio, audio.shape[1])]
        
 
        if audio.shape[1] <= self.num_audio:
            audio = torch.nn.functional.pad(audio, (0, self.num_audio - audio.shape[1]))  # (2, N)

        if self.train:
            if self.labels[rand_ind] == 0:
                splice_audio_start = splice_start * 44100 // 30 
                splice_audio_length = splice_length * self.image_stride * 44100 // 30
                if splice_audio.shape[1] <= splice_audio_length:
                   splice_audio = torch.nn.functional.pad(splice_audio, (0, splice_audio_length - splice_audio.shape[1]))  # (2, N)
                audio[... , splice_audio_start : splice_audio_start + splice_audio_length] = splice_audio[... , : splice_audio_length]

            if random.random() < 0.5:
                C, N = audio.shape
                audio += 0.05 * torch.rand(C, N)
            # if random.random() < 0.5:
            #     audio = (audio.t() * (0.9 + 0.2 * torch.rand(2))).t()

        audio = self.audio_preprocessor(audio) 
        audio = torch.clamp(audio, min = 1e-10).log()    

        if self.train:
            N, C, H, W = images.shape
            if random.random() < 0.5:
                images += 0.05 * torch.randn(N, C, H, W)
            if random.random() < 0.5:
                images[np.random.choice(N, np.random.randint(N//10), replace = False), :, :, :] = 0
            # images = self.image_preprocessor2(images)
            
            audio = audio.unsqueeze(0)
            audio = torchaudio.transforms.TimeMasking(time_mask_param = 100, iid_masks = True, p = 0.5)(audio)
            if random.random() < 0.5:
                audio = torchaudio.transforms.FrequencyMasking(freq_mask_param = 25, iid_masks = True)(audio)
            audio = audio.squeeze(0)
        audio = audio[..., :self.num_images * self.a_i]
        audio = torch.chunk(audio, self.num_images, -1)
        audio = torch.stack(audio, 0)

        return images, audio, torch.from_numpy(np.array(self.labels[index]))

    def __len__(self):
        return len(self.labels)
