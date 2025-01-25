'''
LYCEUM source code.
LYCEUM is a deep learning based Ancient WGS CNV caller tool.
This script, LYCEUM_call.py, is only used to load the weights of pre-trained models
and use them to perform CNV calls.
'''

import os
import torch
import math
torch.cuda.empty_cache()
from tqdm import tqdm
import sys
from performer_pytorch import Performer
import numpy as np
from scipy.stats import mode
from einops import rearrange, repeat
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.preprocessing import sequence
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torchvision.datasets import DatasetFolder
import argparse
import datetime
import pdb
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split
from sklearn.metrics import confusion_matrix

'''
Helper function to print informative messages to the user.
'''

def message(message):
    print("[",datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"]\t", "LYCEUM:\t", message)

'''
Helper function to calculate metrics.
'''
def calculate_metrics(tp_nocall, tp_plus_fp_nocall, tp_duplication, tp_plus_fp_duplication, tp_deletion, tp_plus_fp_deletion, tp_plus_fn_nocall, tp_plus_fn_duplication, tp_plus_fn_deletion):
    nocall_prec = tp_nocall / (tp_plus_fp_nocall+1e-15) 
    dup_prec = tp_duplication / (tp_plus_fp_duplication+1e-15) 
    del_prec = tp_deletion / (tp_plus_fp_deletion+1e-15) 
    nocall_recall = tp_nocall / (tp_plus_fn_nocall+1e-15) 
    dup_recall = tp_duplication / (tp_plus_fn_duplication+1e-15) 
    del_recall = tp_deletion / (tp_plus_fn_deletion+1e-15) 

    return nocall_prec, dup_prec, del_prec, nocall_recall, dup_recall, del_recall


''' 
Perform I/O operations.
'''

description = "LYCEUM is a deep learning based Ancient WGS CNV caller tool. \
            For academic research the software is completely free, although for \
            commercial usage the software is licensed. \n \
            please see ciceklab.cs.bilkent.edu.tr/LYCEUM."

parser = argparse.ArgumentParser(description=description)


required_args = parser.add_argument_group('Required Arguments')
opt_args = parser.add_argument_group("Optional Arguments")

required_args.add_argument("-bs", "--batch_size", help="Batch size to be used in the finetuning.", required=True)


required_args.add_argument("-i", "--input", help="Relative or direct path to input dataset for LYCEUM CNV caller, these are the processed samples for finetuning.", required=True)

required_args.add_argument("-o", "--output", help="Relative or direct output directory path to write LYCEUM output model weights.", required=True)

required_args.add_argument("-s", "--stats_lookup", help="Please provide the path for mean&std lookup file (in .npy format) to normalize. \n \
                                                    These values can be calculated using mean_std_calculator.py file", required=True)


required_args.add_argument("-e", "--epochs", help="Please provide the number of epochs the finetuning will be performed.", required=True)

required_args.add_argument("-lr", "--learning_rate", help="Please provide the learning rate to be used in finetuning", required=True)

required_args.add_argument("-lmp", "--load_model_path", help="Please provide the path for the pretrained model weights to be loaded for finetuning", required=True)

opt_args.add_argument("-g", "--gpu", help="Specify gpu", required=False)



parser.add_argument("-V", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("LYCEUM version 0.1")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    message("Using GPU!")
else:
    message("Using CPU!")

os.makedirs(args.output, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


bs = int(args.batch_size)
N_EPOCHS = int(args.epochs)

LR = float(args.learning_rate)


EXON_SIZE = 1000
PATCH_SIZE = 1
NO_LAYERS = 3 
FEATURE_SIZE = 64
NUM_CLASS = 3

class AddRandom:

    def __call__(self, exon):
        #print(exon, (1,1000))
        #noise = torch.normal(0, 1, size=exon.size())
        mu, sigma = 0, 2 # mean and standard deviation
        noise = np.random.normal(mu, sigma, (1,1000))
        exon[:,:1000] = exon[:,:1000] + noise
        return exon

class MyDataset(DatasetFolder):
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
         
        sample_name = path.split("/")[-1].split("_datapoint_")[0]
        
        try:
            sample_stat = ANCIENT_STAT_VALUES[sample_name]
        except KeyError:
            raise Exception(f"Could not find sample in stats lookup file. \n \
                                Your sample : {str(sample_name)} \n \
                                Samples in stats lookup file : {str(ANCIENT_STAT_VALUES.keys())} \n \
                                ")
                    
        coverage_value = "_".join(path.split("/")[-1].split("_datapoint_")[0].split("_")[1:])
        return sample, target, sample_stat,sample_name, coverage_value
        
    def _find_classes(self, directory: str):
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {}
        for name in classes:
            if name.startswith('nocall'):
                class_to_idx[name] = 0
            elif name.startswith("duplication"):
                class_to_idx[name] = 1
            elif name.startswith("deletion"):
                class_to_idx[name] = 2
        #class_to_idx = {"nocall":0,"duplication":1,"deletion":2}
      
        return classes, class_to_idx



class PositionalEmbedding(nn.Module):
    
        def __init__(self, channels):
            super(PositionalEmbedding, self).__init__()
            inv_freq = 1. / (1000000000 ** (torch.arange(0, channels, 2).float() / channels))
            self.register_buffer('inv_freq', inv_freq)
            
            
        def forward(self, tensor,chrms,strt,ends):
            siz = 1001
            bs = tensor.shape[0]
           
            pos = torch.linspace(strt[0,0].item(), ends[0,0].item(), siz, device=tensor.device).type(self.inv_freq.type()) 

            sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
            emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)

            emb = emb[None,:,:]
            
            
            for i in range(1,bs):
                pos = torch.linspace(strt[i,0].item(), ends[i,0].item(), siz, device=tensor.device).type(self.inv_freq.type()) 
                sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
                emb_i = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
                emb_i = emb_i[None,:,:]
                
                
                emb = torch.cat((emb, emb_i), 0)
            return emb
            
class CNVcaller(nn.Module):
    def __init__(self, exon_size, patch_size, depth,embed_dim,num_class,channels = 1):
        super().__init__()
        assert exon_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (exon_size // patch_size) 
        patch_dim = channels * patch_size 
        self.patch_size = patch_size
        self.exon_size = exon_size
        self.pos_emb = PositionalEmbedding(embed_dim)
        
        self.patch_to_cnn_embedding_v2 = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=1,padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, 3, stride=1,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),          
        )
                

        self.chromosome_token = nn.Parameter(torch.randn(1, 24, embed_dim))
        self.to_cls_token = nn.Identity()
        self.attention = Performer(
            dim = embed_dim,
            depth = depth,
            heads = 8
        )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.mlp_head2 = nn.Linear(embed_dim, num_class)

    

    def forward(self, exon, mask):
        chrs = exon[:,:,-1]
        
        strt = exon[:,:,-2]
        ends = exon[:,:,-3]
        p = self.patch_size

        
        all_ind = list(range(self.exon_size))
        
        indices = torch.tensor(all_ind).to(device)
        exon = torch.index_select(exon, 2, indices).to(device)
        
        all_ind = list(range(self.exon_size + 1))

        indices = torch.tensor(all_ind).to(device)
        mask = torch.index_select(mask, 1, indices).to(device)

        x = self.patch_to_cnn_embedding_v2(exon)
        x = rearrange(x, 'b c (h p1) -> b h (p1 c)', p1 = p)

        batch_size, n, _ = x.shape
        
        crs = self.chromosome_token[:, int(chrs[0,0].item()-1): int(chrs[0,0].item()), :]
       
        for i in range(1,batch_size):
            crs_ = self.chromosome_token[:, int(chrs[i,0].item()-1): int(chrs[i,0].item()), :]
            
            crs = torch.cat((crs, crs_), 0)
        
        x = torch.cat((crs, x), dim=1)
        
        x += self.pos_emb(x,chrs,strt,ends)  
        x = self.attention(x,input_mask = mask)
        x = self.to_cls_token(x[:, 0])
    
        y = self.mlp_head(x)

        z = self.mlp_head2(y)
        return z



def train_func(epoch,model):
    my_dataset = MyDataset(args.input, loader=np.load, extensions = (".npy",".npz"))

    sub_train_ = DataLoader(my_dataset, batch_size = bs, shuffle=True) 

    data = sub_train_
    
    train_loss, train_acc = 0, 0

    tp_tn_fp_fn = 0
    tp_nocall = 0
    tp_duplication = 0
    tp_plus_fp_nocall = 0
    tp_plus_fp_duplication = 0 
    tp_plus_fp_deletion = 0

    tp_plus_fn_nocall = 0
    tp_plus_fn_duplication = 0
    tp_plus_fn_deletion = 0
    tp_deletion = 0

    model.train()
    for i, (exons, labels, sample_stat,sample_name, coverage_value) in enumerate(data):
        ind = torch.arange(labels.size(0))
        optimizer.zero_grad()
        
        labels = labels.long()
        exons, labels = exons.to(device),  labels.to(device)
        
        exons = exons.float()
        mask = torch.logical_not(exons == -1)
        
        mean_vals = torch.unsqueeze(torch.unsqueeze(sample_stat["mean"], 1),1).to(device)
        std_vals = torch.unsqueeze(torch.unsqueeze(sample_stat["std"], 1),1).to(device)
        exons[:,:,:1000] -= mean_vals
        exons[:,:,:1000] /=  std_vals
        
        mask = torch.squeeze(mask)
        real_mask = torch.ones(labels.size(0),1006, dtype=torch.bool).to(device)
        
        real_mask[:,1:] = mask
       
        output1 = model(exons, real_mask)
        log_outputs = torch.log_softmax(output1, 1)
       
     
       
        loss1 = CEloss( log_outputs, labels) 

  
        train_loss += loss1.item()
        loss1.backward()
    
       
        optimizer.step()
        
        
        _, predicted = torch.max(output1.data, 1)
   
     
        tp_nocall += (torch.logical_and(predicted == labels,predicted == 0)).sum().item() 
        tp_duplication += (torch.logical_and(predicted == labels,predicted == 1)).sum().item() 
        tp_deletion += (torch.logical_and(predicted == labels,predicted == 2)).sum().item()

        train_acc += (predicted == labels).sum().item()
        tp_plus_fp_nocall += (predicted == 0).sum().item()
        tp_plus_fp_duplication += (predicted == 1).sum().item()
        tp_plus_fp_deletion += (predicted == 2).sum().item()
   
        
        tp_plus_fn_nocall += (labels == 0).sum().item()
        tp_plus_fn_duplication += (labels == 1).sum().item()
        tp_plus_fn_deletion += (labels == 2).sum().item()
        tp_tn_fp_fn += labels.size(0)

        if i % 100 == 0 and i > 0:
            nocall_prec, dup_prec, del_prec, nocall_recall, dup_recall, del_recall = calculate_metrics(tp_nocall, tp_plus_fp_nocall, tp_duplication, tp_plus_fp_duplication, tp_deletion, tp_plus_fp_deletion, tp_plus_fn_nocall, tp_plus_fn_duplication, tp_plus_fn_deletion)    
            message(f"Model weights are saved for Epoch: {round(epoch+(((i+1)*bs)/len(my_dataset)),2)}")
            torch.save(model.state_dict(), os.path.join(args.output, f"lyceum_ft64_depth3_exonsize1000_patchsize1_lr{str(LR)}.pt"))
            message(f'Batch no: {i}\tLoss: {train_loss / (i+1):.4f}(train)\t|\tNocall_prec: {nocall_prec * 100:.1f}%(train)|\tDup_prec: {dup_prec * 100:.1f}%(train)|\tDel_prec: {del_prec * 100:.1f}%(train)|\tNocall_recall: {nocall_recall * 100:.1f}%(train)|\tDup_recall: {dup_recall * 100:.1f}%(train)|\tDel_recall: {del_recall * 100:.1f}%(train)')
    
    nocall_prec, dup_prec, del_prec, nocall_recall, dup_recall, del_recall = calculate_metrics(tp_nocall, tp_plus_fp_nocall, tp_duplication, tp_plus_fp_duplication, tp_deletion, tp_plus_fp_deletion, tp_plus_fn_nocall, tp_plus_fn_duplication, tp_plus_fn_deletion)
    
    return train_loss / len(data), ( train_acc / tp_tn_fp_fn), nocall_prec,dup_prec,del_prec, nocall_recall, dup_recall,del_recall



ANCIENT_STAT_VALUES = np.load(args.stats_lookup,allow_pickle=True).item()

## Set model
model = CNVcaller( EXON_SIZE, PATCH_SIZE, NO_LAYERS, FEATURE_SIZE, NUM_CLASS)
model.load_state_dict(torch.load(args.load_model_path, map_location=device))
model.to(device)
print("number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

min_valid_loss = float('inf')

optimizer = torch.optim.Adam(model.parameters(),lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,N_EPOCHS)

CEloss = nn.NLLLoss()

message("Starting training...")

#### Fine-tuning

message(f"Fine-tuning started\n")
os.makedirs(args.output ,exist_ok=True)
for epoch in range(N_EPOCHS): 
    start_time = time.time()
    train_loss, train_acc,nocall_prec,dup_prec,del_prec, nocall_recall, dup_recall,del_recall = train_func(epoch,model)
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    message(f"Model weights are saved for Epoch: {epoch+1}")
    torch.save(model.state_dict(), os.path.join(args.output, f"lyceum_ancient_ft64_depth3_exonsize1000_patchsize1_e_{str(epoch)}.pt"))
    
    scheduler.step()
    message("Epoch: %d | time in %d minutes, %d seconds" %( epoch, mins, secs))
    message(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)|\tNocall_prec: {nocall_prec * 100:.1f}%(train)|\tDup_prec: {dup_prec * 100:.1f}%(train)|\tDel_prec: {del_prec * 100:.1f}%(train)|\tNocall_recall: {nocall_recall * 100:.1f}%(train)|\tDup_recall: {dup_recall * 100:.1f}%(train)|\tDel_recall: {del_recall * 100:.1f}%(train)')
    
    
message(f"Fine-tuning ended")



   
