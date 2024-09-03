import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm
import pandas as pd
import random
import argparse, os
import torchvision
import torchvision.transforms as transforms
import torchaudio

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
        
class dataset(object):
    def __init__(self, label_list_path, sample_stride = 1, sample_n_frames = 90, n_fft = 1024, win_length = 1024, hop_length = 147, a_i = 10, **kwargs):
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
        
        self.image_preprocessor = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
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
        images = self.image_preprocessor(images)   # (N,C,H,W)

        start_idx = start_idx * 44100 // 30
        audio = audio[..., start_idx : min(start_idx + self.num_audio, audio.shape[1])]
        if audio.shape[1] <= self.num_audio:
            audio = torch.nn.functional.pad(audio, (0, self.num_audio - audio.shape[1]))  # (2, N)
        audio = self.audio_preprocessor(audio) 
        audio = torch.clamp(audio, min = 1e-10).log()    
        audio = audio[..., :self.num_images * self.a_i]
        audio = torch.chunk(audio, self.num_images, -1)
        audio = torch.stack(audio, 0)

        return self.video_names[index], images, audio

    def __len__(self):
        return len(self.labels)
    
def main(model_path, test_path, save_path):
    model = DetectModel(**vars(args))
    self_state = model.state_dict()
    print('loading model...')
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

    print('model inferring...')
    valset = dataset(label_list_path = test_path)
    valloader = torch.utils.data.DataLoader(valset, batch_size = 8, shuffle = False, num_workers = 16)
    preds = []
    for _, (path, image, audio) in tqdm.tqdm(enumerate(valloader)):
        image = image.cuda()
        audio = audio.cuda()
        with torch.no_grad():
            output = model(image, audio)
        output = F.softmax(output, dim = 1).detach().cpu().numpy()[:,1]
        for i in range(len(path)):
            preds.append([path[i], output[i]])
    df_result = pd.DataFrame(preds, columns=["video_name", "y_pred"])
    df_result.to_csv(save_path, index=False, header=True)
    return 
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "DeepFake videoo")
    parser.add_argument('--model_path',      type=str,   default='./exps/detectmodel/epoch_0.model',       help='Model checkpoint path')
    parser.add_argument('--test_path',      type=str,   default='/data/ai_security/dataset/multiFFDV/phase1/valset_label.txt',       help='Path of test file')
    parser.add_argument('--save_path', type=str, default='./exps/detectmodel/val_0.csv', help='Path of result')
    args = parser.parse_args()
    main(args.model_path, args.test_path, args.save_path)
    print('done!')