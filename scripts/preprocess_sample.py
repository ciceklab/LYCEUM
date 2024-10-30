'''
LYCEUM samples preprocessing source code.
This script generates and processes the samples to perform CNV calling.
'''
import numpy as np
import os
from os import listdir
import pdb
import csv
import pandas as pd
import time
from tqdm import tqdm
from multiprocessing.pool import Pool
import argparse


description = "LYCEUM is a deep learning based Ancient WGS CNV caller tool. \
            For academic research the software is completely free, although for \
            commercial usage the software is licensed. \n \
            please see ciceklab.cs.bilkent.edu.tr/LYCEUMlicenceblablabla."

parser = argparse.ArgumentParser(description=description)

required_args = parser.add_argument_group('Required Arguments')

required_args.add_argument("-rd", "--readdepth", help="Please provide the exon-wise readdepth path.", required=True)

required_args.add_argument("-o", "--output", help="Please provide the output path for the preprocessing.", required=True)

required_args.add_argument("-t", "--target", help="Please provide the path of exon target file.", required=True)

parser.add_argument("-V", "--version", help="show program version", action="store_true")
args = parser.parse_args()

exon_wise_readdepths_path = args.readdepth
output_path = args.output
target_list_path = args.target

os.makedirs(args.output, exist_ok=True)

exon_wise_readdepth_files = listdir(exon_wise_readdepths_path)
for file_name in tqdm(exon_wise_readdepth_files):

    labeled_data = []
    sample_name = file_name.split(".bam.txt")[0].split(".")[0].split("_")[0] 
    output_name = file_name.split(".bam.txt")[0]
    target_data = pd.read_csv(target_list_path, sep="\t", header=None).values
    read_depth_data = pd.read_csv(os.path.join(exon_wise_readdepths_path, file_name), sep="\t")
    read_depth_data["REF"] = read_depth_data["REF"].astype(str)
    read_depth_data = read_depth_data.values
    
    chromosomes = np.unique(target_data[:,0])
    
    for chr_ in tqdm(chromosomes):
        chr_formatted = chr_
        char_array = np.array(read_depth_data[:,0]).astype(str)
        cond_1 = char_array== chr_
        
        if not np.any(cond_1):
            continue
        
        cond_3 = target_data[:,0] == chr_
        
        rd_d = read_depth_data[cond_1][:,2]
        rd_data_loc = read_depth_data[cond_1][:,1].astype(int) 
        
        ct= 0
        target_data_cond = target_data[cond_3]
        for i in range(target_data_cond.shape[0]):
        
            target_chr = target_data_cond[i,0]
            target_st = target_data_cond[i,1]
            target_end = target_data_cond[i,2]
            
            range_mask = (rd_data_loc<target_end) & (rd_data_loc >= target_st)
           
            loc_with_reads_within_interval = rd_data_loc[range_mask]
            
            rd_seq = np.zeros(target_end-target_st)
            
            rd_seq[loc_with_reads_within_interval-target_st] = rd_d[range_mask]            
       
            
            data_point = []
            data_point.append(sample_name)
            data_point.append(target_chr)
            data_point.append(target_st)
            data_point.append(target_end)
            data_point.append(rd_seq)
                                    
            labeled_data.append(data_point)
                            
    labeled_data = np.asarray(labeled_data)
    np.save(os.path.join(output_path,output_name+"_labeled_data.npy"), labeled_data)

    



        