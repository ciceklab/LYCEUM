import os
import numpy as np
import csv
import sys
import pdb 
from tqdm import tqdm
import argparse


description = "It generates a mean and standard deviation lookup file for the target ancient samples."

parser = argparse.ArgumentParser(description=description)

required_args = parser.add_argument_group('Required Arguments')

required_args.add_argument("-p", "--processed_samples", help="Please provide the path for processed samples.", required=True)
required_args.add_argument("-n", "--setname", help="Please provide a name for the ancient set.", required=True)
required_args.add_argument("-o", "--output", help="Please provide the output path for the mean_std dict file.", required=True)

parser.add_argument("-V", "--version", help="show program version", action="store_true")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

mean_std_dict = {}
def calculate_mean_std(input_directory):
    for file_name in tqdm(os.listdir(input_directory)):
        if file_name.endswith(".npy"):
            file_path = os.path.join(input_directory, file_name)
            
            data = np.load(file_path, allow_pickle=True)
            
            merged_array = []
            
            for row in data:
                if isinstance(row[4], np.ndarray):
                    merged_array.extend(row[4])

            merged_array = np.array(merged_array)

            mean_value = np.mean(merged_array)
            std_value = np.std(merged_array)
            sample_name = file_name.split("_labeled_data.npy")[0]
            mean_std_dict[f"{sample_name}"] = {'mean': mean_value, 'std': std_value}
    return mean_std_dict


if __name__ == "__main__":
    mean_std_dict = calculate_mean_std(args.processed_samples)
    np.save(os.path.join(args.output,args.setname)+"_stats_lookup.npy", mean_std_dict)
