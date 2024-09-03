import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm
import pandas as pd
import argparse, os
import torchvision
import torchvision.transforms as transforms
import torchaudio
import math
from einops import rearrange

from utils.DataLoader import dataset
from models.DetectModel import DetectModel

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

class Inferencer(object):
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.image_preprocessor = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.audio_preprocessor = nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.Spectrogram(n_fft = 1024, win_length = 1024, hop_length = 147, window_fn=torch.hamming_window, power=2),
            )

    def load_model(self, model_path):
        model = DetectModel()
        self_state = model.state_dict()
        loaded_state = torch.load(model_path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("speaker_encoder.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

        model.load_state_dict(self_state)
        model.eval().cuda()
        return model

    def eval_embedding(self, video_path, sample_n_frames = 90):
        video, audio, info = torchvision.io.read_video(video_path, pts_unit="sec")

        video_length = len(video)
        images = video.permute(0,3,1,2).contiguous()  
        images = images / 255.
        if video_length <= 30:
            num_clip = 1
        else:
            num_clip = math.ceil((video_length-30)/sample_n_frames)
        if video_length < num_clip * sample_n_frames:
            images = torch.nn.functional.pad(images, (0, 0, 0, 0, 0, 0, 0, num_clip * sample_n_frames - video_length)) 
        images = self.image_preprocessor(images[:num_clip * sample_n_frames])
        images = torch.chunk(images, num_clip, 0)
        images = torch.stack(images, 0)
            
        max_audio = num_clip * sample_n_frames * 44100 // 30
        if audio.shape[1] < max_audio:
            audio = torch.nn.functional.pad(audio, (0, max_audio - audio.shape[1]))
        audio = rearrange(audio[..., :max_audio], 'n (b d) -> (b n) d', b = num_clip)
        audio = self.audio_preprocessor(audio) 
        audio = rearrange(audio, '(b n) h w -> b n h w', b = num_clip)
        audio = torch.clamp(audio, min = 1e-10).log()    
        audio = audio[..., :sample_n_frames * 10]
        audio = torch.chunk(audio, sample_n_frames, -1)
        audio = torch.stack(audio, 1)

        with torch.no_grad():
            outputs = []
            for i in range(math.ceil(images.shape[0]/8)):
                output = self.model.forward(images[8 * i : min(8 * i + 8, images.shape[0])].cuda(), audio[8 * i : min(8 * i + 8, audio.shape[0])].cuda())
                outputs = outputs + list(F.softmax(output, dim = 1).detach().cpu().numpy()[:,1])
        return min(outputs)
       

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "DeepFake video")
    parser.add_argument('--model_path',      type=str,   default='./exps/detectmodel/epoch_0.model',       help='Model checkpoint path')
    parser.add_argument('--test_path',      type=str,   default='/data/ai_security/dataset/multiFFDV/phase1/trainset/dac6604f700f216e865139ad6842a28b.mp4',       help='Path of test file')
    args = parser.parse_args()
    infer = Inferencer(model_path = args.model_path)
    output = infer.eval_embedding(args.test_path)
    print(output)