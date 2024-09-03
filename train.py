import numpy as np
import torch
import tqdm
import argparse
import torch.nn as nn
import sys, os, warnings
from sklearn import metrics
import time

from utils.DataLoader import dataset
from models.DetectModel import DetectModel

parser = argparse.ArgumentParser()
parser.add_argument('--device', type = str, default = 'cuda', help = 'Device training on ')
## Training Settings
parser.add_argument('--num_frames', type = int, default = 90, help = 'Duration of the image segments')
parser.add_argument('--stride', type = int, default = 1, help = 'Sample stride of images')
parser.add_argument('--max_epoch', type = int, default = 80, help = 'Maximum number of epochs')
parser.add_argument('--batch_size', type = int, default=8, help = 'Batch size')
parser.add_argument('--num_workers', type = int, default = 16, help = 'Number of loader threads')
parser.add_argument('--test_step', type = int, default = 1, help = 'Test and save every [test_step] epochs')
parser.add_argument('--lr', type = float, default = 1e-4, help = 'Learning rate')
parser.add_argument("--T_max", type = int, default = 1000, help = 'Maximum number of iterations to init lr')
parser.add_argument("--eta_min", type = int, default = 1e-5, help = 'Minimum learning rate')

## Training and evaluation path/lists, save path
parser.add_argument('--train_label', type = str, default = "/data/ai_security/dataset/multiFFDV/phase1/trainset_label.txt", help = 'The path of the training label list')
parser.add_argument('--val_label', type = str, default = "/data/ai_security/dataset/multiFFDV/phase1/valset_label.txt", help = 'The path of the evaluation list')
parser.add_argument('--save_path', type=str, default = "detectmodel", help = 'Path to save models')
parser.add_argument('--init_model', type = str, default = "", help = 'Path of the initial_model')

## Initialization
args = parser.parse_args()
warnings.simplefilter("ignore")

def metrics_scores(output, target):
    output = output.detach().cpu().numpy().argmax(axis=1)
    target = target.detach().cpu().numpy()
    
    accuracy = metrics.accuracy_score(target, output)
    recall = metrics.recall_score(target, output)
    precision = metrics.precision_score(target, output)
    
    return accuracy*100, recall*100, precision*100

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        if self.weight is not None:
            weight = self.weight.gather(0,targets. view(-1))
        else:
            weight = 1.0

        logpt = torch.nn.functional.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt  = logpt.gather(1,targets.view(-1,1))
        pt = pt.gather(1,targets.view(-1,1))
        loss = -((1 - pt) ** self.gamma) * logpt
        loss = weight * loss.t()
        return loss.mean()

trainset = dataset(label_list_path = args.train_label, sample_stride = args.stride, sample_n_frames = args.num_frames, train = True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
valset = dataset(label_list_path = args.val_label, sample_stride = args.stride, sample_n_frames = args.num_frames)
valloader = torch.utils.data.DataLoader(valset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

model = DetectModel(**vars(args))
if args.init_model != '':
    model.load_state_dict(torch.load(args.init_model))
model = model.to(args.device)

criterion = FocalLoss(gamma=2, weight=torch.FloatTensor([3.0, 1.0]).to(args.device))
optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.T_max, eta_min = args.eta_min)

save_path = './exps/' + args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
info_path = os.path.join(save_path, 'info.txt')
info_file = open(info_path, "a+")

start = time.time()
for epoch in range(1):
    model.train()
    index, acc, loss, recall, prec = 0, 0, 0, 0, 0
    for i, (image, audio, label) in enumerate(trainloader, start = 1):
        image = image.to(args.device)
        audio = audio.to(args.device)
        label = label.to(args.device)
        output = model(image, audio)
        loss_t = criterion(output, label)

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        acc_t, recall_t, prec_t = metrics_scores(output, label)
        index += len(label)
        acc += acc_t
        recall += recall_t
        prec += prec_t
        loss += loss_t.detach().cpu().numpy()
        sys.stderr.write(time.strftime("%H:%M:%S  ", time.gmtime(time.time()-start)) + \
        "Epoch[%2d] Lr: %5f, Training: [%2d|%2d] %.2f%%, " %(epoch, lr, i, trainloader.__len__(), 100 * (i / trainloader.__len__())) + \
        "Loss: %.5f, Acc: %2.2f%%, Recall: %2.2f%%, Prec: %2.2f%%\r" %(loss_t.detach().cpu().numpy(), acc/index*len(label), recall/index*len(label), prec/index*len(label)))
        sys.stderr.flush()

        if i % 10 == 0:
            scheduler.step()

    sys.stdout.write("\n")
    torch.save(model.state_dict(), save_path + '/epoch_' + str(epoch) + '.model' )

    model.eval()
    preds = []
    labels = []
    for _, (image, audio, label) in tqdm.tqdm(enumerate(valloader)):
        image = image.to(args.device)
        audio = audio.to(args.device)
        with torch.no_grad():
            output = model(image, audio)
        preds.append(output)
        labels.append(label)
    val_acc, val_recall, val_prec = metrics_scores(torch.cat(preds,0), torch.cat(labels,0))

    info_file.write("%d epoch, loss %.5f, Acc %2.2f%%, Recall %2.2f%%, Prec %2.2f%%, Val_acc %2.2f%%, Val_recall %2.2f%%, Val_prec  %2.2f%%\n"%(epoch, loss/(i), acc/index*len(label), recall/index*len(label), prec/index*len(label), val_acc, val_recall, val_prec))
    info_file.flush()
    

