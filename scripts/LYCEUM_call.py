'''
LYCEUM source code.
LYCEUM is a deep learning based WES CNV caller tool.
This script, LYCEUM_call.py, is only used to load the weights of pre-trained models
and use them to perform CNV calls.
'''
import numpy as np
from sklearn.metrics import confusion_matrix as cm
from tensorflow.keras.preprocessing import sequence
import torch
from performer_pytorch import Performer
from einops import rearrange, repeat
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader,Dataset
import pandas as pd
import os
from itertools import groupby
from tqdm import tqdm
import argparse
import datetime
import pdb
from collections import Counter


'''
Helper function to print informative messages to the user.
'''
def message(message):
    print("[",datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"]\t", "LYCEUM:\t", message)


'''
Helper function to convert string chromosome to integer.
'''
def convert_chr_to_integer(chr_list):
    for i in range(len(chr_list)):
            crs = chr_list[i]
            if(len(crs) == 4):
                if(crs[3] == "Y"):
                    crs = 23
                elif (crs[3] == "X"):
                    crs = 24
                else:
                    crs = int(crs[3])
            elif(len(crs) == 5):
                crs = int(crs[3:5])
            chr_list[i] = crs

cur_dirname = os.path.dirname(__file__)

''' 
Perform I/O operations.
'''

description = "LYCEUM is a deep learning based WES CNV caller tool. \
            For academic research the software is completely free, although for \
            commercial usage the software is licensed. \n \
            please see ciceklab.cs.bilkent.edu.tr/LYCEUMlicenceblablabla."

parser = argparse.ArgumentParser(description=description)

'''
Required arguments group:
(i) -m, pretrained models of the paper, one of the following: (1) lyceum, 
(ii) -i, input data path comprised of WES samples with read depth data.
(iii) -o, relative or direct output directory path to write LYCEUM output file.
(v) -c, Depending on the level of resolution you desire, choose one of the options: (1) exonlevel, (2) genelevel
(vi) -t, Path of exon target file.
(vii) -n, The path for mean&std stats of read depth values.
'''

required_args = parser.add_argument_group('Required Arguments')
opt_args = parser.add_argument_group("Optional Arguments")

required_args.add_argument("-m", "--model", help="If you want to use pretrained LYCEUM weights choose one of the options: ", required=True)

required_args.add_argument("-bs", "--batch_size", help="Batch size to be used in the finetuning.", required=True)

required_args.add_argument("-i", "--input", help="Relative or direct path to input files for LYCEUM CNV caller, these are the processed samples.", required=True)

required_args.add_argument("-o", "--output", help="Relative or direct output directory path to write LYCEUM output file.", required=True)

required_args.add_argument("-c", "--cnv", help="Depending on the level of resolution you desire, choose one of the options: \n \
                                                (i) exonlevel, (ii) genelevel", required=True)

required_args.add_argument("-n", "--normalize", help="Please provide the path for mean&std stats of read depth values (in .npy format) to normalize. \n \
#                                                    These values are obtained precalculated from ancient samples.", required=True)


opt_args.add_argument("-g", "--gpu", help="Specify gpu", required=False)

'''
Optional arguments group:
-v or --version, version check
-h or --help, help
-g or --gpu, specify gpu
-
'''

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

try:
    os.mkdir(os.path.join(args.output,"tmp/"))

except OSError:
    print ("Creation of the directory failed")
else:
    print ("Successfully created the directory")
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)

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
        

        indices = torch.tensor(all_ind).to(exon.device)
        exon = torch.index_select(exon, 2, indices).to(exon.device)
        
        all_ind = list(range(self.exon_size + 1))
        
        
        
        indices = torch.tensor(all_ind).to(exon.device)
        mask = torch.index_select(mask, 1, indices).to(exon.device)

        x = rearrange(exon, 'b c (h p1) -> b h (p1 c)', p1 = p)
        x = self.patch_to_embedding(x)
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


def call_cnv_regions(all_samples_names):
    for sample_name in tqdm(all_samples_names):

        sampledata = np.load(os.path.join(args.input, sample_name+"_labeled_data.npy"), allow_pickle=True)    
        message(f"calling sample: {sample_name}")

        sampnames_data = []
        chrs_data = []
        readdepths_data = []
        start_inds_data = []
        end_inds_data = []
        wgscalls_data = []

        temp_sampnames = sampledata[:,0]
        temp_chrs = sampledata[:,1]
        temp_start_inds = sampledata[:,2]
        temp_end_inds = sampledata[:,3]
        temp_readdepths = sampledata[:,4]

        for i in range(len(temp_chrs)):
            
            crs = temp_chrs[i]
            if(len(crs) == 4):
                if(crs[3] == "Y"):
                    crs = 23
                elif (crs[3] == "X"):
                    crs = 24
                else:
                    crs = int(crs[3])
            elif(len(crs) == 5):
                crs = int(crs[3:5])

            temp_chrs[i] = crs
            temp_readdepths[i] = list(temp_readdepths[i])
            arr = np.array(temp_readdepths[i],dtype=float)

            temp_readdepths[i].insert(len(temp_readdepths[i]), 0)
            temp_readdepths[i].insert(len(temp_readdepths[i]), 0)
        
            temp_readdepths[i].insert(len(temp_readdepths[i]), temp_end_inds[i])
            temp_readdepths[i].insert(len(temp_readdepths[i]), temp_start_inds[i]) 

            temp_readdepths[i].insert(len(temp_readdepths[i]), crs) 
            
        sampnames_data.extend(temp_sampnames)
        chrs_data.extend(temp_chrs)
        readdepths_data.extend(temp_readdepths)
        start_inds_data.extend(temp_start_inds)
        end_inds_data.extend(temp_end_inds)

        lens = [len(v) for v in readdepths_data]

        lengthfilter = [True if v < 1006  else False for v in lens]

        sampnames_data = np.asarray(sampnames_data)[lengthfilter]
        chrs_data = np.asarray(chrs_data)[lengthfilter]
        readdepths_data = np.asarray(readdepths_data)[lengthfilter]
        start_inds_data = np.asarray(start_inds_data)[lengthfilter]
        end_inds_data = np.asarray(end_inds_data)[lengthfilter]

        readdepths_data = np.asarray([np.asarray(k) for k in readdepths_data])
        readdepths_data = sequence.pad_sequences(readdepths_data, maxlen=1005,dtype=np.float32,value=-1)
        readdepths_data = readdepths_data[ :, None, :]

        stat_values = ANCIENT_STAT_VALUES[f"{sample_name}"]
        means_ = float(stat_values["mean"])
        stds_ = float(stat_values["std"])

        test_x = torch.FloatTensor(readdepths_data)
        test_x = TensorDataset(test_x)
        x_test = DataLoader(test_x, batch_size=int(args.batch_size))

        allpreds = []
        allExonMeans_data= []

        for exons in tqdm(x_test):
        
            exons = exons[0].to(device)

            mask = torch.logical_not(exons == -1)
            
            ###
            exon_means = ((exons[:,:,:1000].sum(axis=2)+(~mask[:,:,:1000]).sum(axis=2))  / mask.sum(axis=2)).squeeze()
            ###

            exons[:,:,:1000] -= means_
            exons[:,:,:1000] /=  stds_

            mask = torch.squeeze(mask)
            real_mask = torch.ones(exons.size(0),1006, dtype=torch.bool).to(device)
            real_mask[:,1:] = mask

            output1 = model(exons,real_mask)
            
            _, predicted = torch.max(output1.data, 1)

            
            preds = list(predicted.cpu().numpy().astype(np.int64))
            exon_means_list = list(exon_means.cpu().numpy().astype(np.float))
            allpreds.extend(preds)
            allExonMeans_data.extend(exon_means_list)            

        chrs_data = chrs_data.astype(int)
        allpreds = np.array(allpreds)
        allExonMeans_data = np.array(allExonMeans_data)
        for j in tqdm(range(1,25)):
            indices = chrs_data == j

            predictions = allpreds[indices]
            start_inds = start_inds_data[indices]
            end_inds = end_inds_data[indices]
            allExonMeans = allExonMeans_data[indices]

            sorted_ind = np.argsort(start_inds)
            
            predictions = predictions[sorted_ind]
            end_inds = end_inds[sorted_ind]
            start_inds = start_inds[sorted_ind]
            allExonMeans  = allExonMeans[sorted_ind]
            
            
            chr_ = "chr"
            if j < 23:
                chr_ += str(j)
            elif j == 23:
                chr_ += "Y"
            elif j == 24:
                chr_ += "X"
        
            for k_ in range(len(end_inds)):       
                f = open(os.path.join(os.path.join(args.output,"tmp"), sample_name + ".csv"), "a")
                f.write(chr_ + "," + str(start_inds[k_]) + "," + str(end_inds[k_]) + ","+ str(int(predictions[k_]))+","+str(allExonMeans[k_]) + "\n")
                f.close()



def call_cnv_regions_without_readDepth(all_samples_names):
    COVERAGE_THRESHOLD =  0.001   
    
    for sample_name in tqdm(all_samples_names):
        message(f"Processing sample: {sample_name}")
        lyceum_calls = pd.read_csv(os.path.join(os.path.join(args.output,"tmp"), sample_name + ".csv",),sep=",",header=None)
        target_data = pd.read_csv("../exon_region_hg38_hg19_lookup_withGene.csv", sep="\t")

        lyceum_calls = target_data.merge(lyceum_calls, how='left', right_on=[0,1,2],left_on=["chrom_hg38","start_hg38","end_hg38"])
        lyceum_calls = lyceum_calls.drop([0, 1, 2,"chrom_hg19","start_hg19","end_hg19","hg38_ind"], axis=1)
        lyceum_calls = lyceum_calls.rename(columns={3:"prediction",4:"rd_mean"})
        lyceum_calls = lyceum_calls.drop_duplicates()
        
        lyceum_calls.loc[lyceum_calls["prediction"] == 0,"prediction"] = "<NO-CALL>"
        lyceum_calls.loc[lyceum_calls["prediction"] == 1,"prediction"] = "<DUP>"
        lyceum_calls.loc[lyceum_calls["prediction"] == 2,"prediction"] = "<DEL>"
        
        lyceum_calls.loc[lyceum_calls.isna()["prediction"],"prediction"] = "<NO-CALL>" # TODO ?
        

        poor_read_mask = (lyceum_calls["rd_mean"] < COVERAGE_THRESHOLD) | ((lyceum_calls["end_hg38"])-(lyceum_calls["start_hg38"])>1000)
        read_poor_regions = lyceum_calls[poor_read_mask]
        read_rich_regions = lyceum_calls[~poor_read_mask]
        
        #MAJOR VOTING OF EXONS WITHIN THE RANGE
        chroms = read_poor_regions["chrom_hg38"].unique()
        for chrom in tqdm(chroms):
            read_poor_regions_filtered = read_poor_regions[read_poor_regions['chrom_hg38']==chrom]
            read_rich_regions_filtered = read_rich_regions[read_rich_regions['chrom_hg38']==chrom].reset_index(drop=True)
            for i,row in read_poor_regions_filtered.iterrows():    
                downStream_loc = int(max(0,row["start_hg38"] - 500))
                upStream_loc = int(row["end_hg38"] + 500)
                matching_exons_downStream = read_rich_regions_filtered[(downStream_loc <= read_rich_regions_filtered['end_hg38']) & ( row["start_hg38"] >= read_rich_regions_filtered['start_hg38'])  & (chrom == row['chrom_hg38'])]
                matching_exons_upStream = read_rich_regions_filtered[(row["end_hg38"] <= read_rich_regions_filtered['end_hg38']) & ( upStream_loc >= read_rich_regions_filtered['start_hg38'])  & (read_rich_regions_filtered['chrom_hg38'] == row['chrom_hg38'])]
                if((len(matching_exons_downStream)!=0) | (len(matching_exons_upStream)!=0)):
                    lyceum_calls.loc[i,"prediction"] = Counter(matching_exons_downStream["prediction"].to_list()+matching_exons_upStream["prediction"].to_list()).most_common()[0][0]        

        if(args.cnv == "exonlevel"):
            f = open(os.path.join(args.output, sample_name + ".csv"), "a")
            f.write("Sample Name" + "\t" +"Chromosome" + "\t" + "CNV Start Index" + "\t" + "CNV End Index" + "\t" + "lyceum Prediction" + "\n")    

            for i,row in lyceum_calls.iterrows():
                f.write(sample_name + "\t" + row["chrom_hg38"] + "\t" + str(row["start_hg38"]) + "\t" + str(row["end_hg38"]) + "\t"+ (row["prediction"])+ "\n")
            f.close()
            os.remove(os.path.join(os.path.join(args.output,"tmp"), sample_name + ".csv",))

            
        elif(args.cnv == "genelevel"):
            lyceum_gene_calls = lyceum_calls.dropna(subset=["gene_name"]).reset_index(drop=True)
            lyceum_gene_calls.loc[lyceum_gene_calls.isna()["prediction"],"prediction"] = "<NO-CALL>"
            lyceum_gene_calls.rename(columns = {'prediction':'exon_predictions'}, inplace = True)
            
            lyceum_gene_calls_grouped = pd.DataFrame(lyceum_gene_calls.groupby(["gene_name","chrom_hg38"],group_keys=True)['exon_predictions'].apply(list).apply(Counter))
            lyceum_gene_calls_grouped["gene_prediction"]=pd.NA
            lyceum_gene_calls_grouped = lyceum_gene_calls_grouped.reset_index()
            
            #GENE-LEVEL MAJORITY VOTING OF GENES
            for (ind), row in lyceum_gene_calls_grouped.iterrows():
                pred = row["exon_predictions"].copy()
                del pred["<NO-CALL>"]
                if(len(pred)==0):
                    lyceum_gene_calls_grouped.loc[ind,"gene_prediction"] = "<NO-CALL>"
                else:
                    #lyceum_gene_calls_grouped.loc[ind,"exon_predictions"] = pred.most_common()
                    exon_count_with_common_sv = pred.most_common(1)[0][1]
                    if(exon_count_with_common_sv>0):
                        lyceum_gene_calls_grouped.loc[ind,"gene_prediction"] = (pred.most_common(1)[0][0])
                    else:
                        lyceum_gene_calls_grouped.loc[ind,"gene_prediction"] = "<NO-CALL>"  
                        
            f = open(os.path.join(args.output, sample_name + ".csv"), "a")
            f.write("Sample Name" + "\t" +"Chromosome" + "\t" + "Gene Name" + "\t" + "lyceum Prediction" + "\n")    

            for i,row in lyceum_gene_calls_grouped.iterrows():
                f.write(sample_name + "\t" + row["chrom_hg38"] + "\t" + str(row["gene_name"])+ "\t"+ (row["gene_prediction"])+ "\n")
            f.close()
            os.remove(os.path.join(os.path.join(args.output,"tmp"), sample_name  + ".csv",))
                                                                      
            
        else:
            os.remove(os.path.join(os.path.join(args.output,"tmp"), sample_name + ".csv"))
            raise Exception("Invalid CNV parameter. It should be either exonLevel or geneLevel")
        
    
    return
    

model = CNVcaller(1000, 1, 3, 192, 3)

if args.model == "lyceum":
    model.load_state_dict(torch.load(os.path.join(cur_dirname, "../models/lyceum_model.pt"), map_location=device))
else:
    message("Something Wrong.")
    
model.eval()
model = model.to(device)


ANCIENT_STAT_VALUES = np.load(args.normalize,allow_pickle=True).item()


input_files = os.listdir(args.input)
all_samples_names = [file.split("_labeled_data.npy")[0] for file in input_files] #NOTE

message("Calling for CNV regions...")
call_cnv_regions(all_samples_names)
    
message("Calling for regions without read depth information..")
call_cnv_regions_without_readDepth(all_samples_names)
    
os.rmdir(os.path.join(args.output,"tmp"))     