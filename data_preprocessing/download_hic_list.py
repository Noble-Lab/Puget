# The scripts are adapted from https://github.com/Noble-Lab/HiCFoundation_paper/blob/main/notebooks/pretrain_data.ipynb
import pandas as pd
import os
import argparse

def parse_hiclist(hiclist_path):
    hic_list = pd.read_csv(hiclist_path, names = ["Hi-C Accession"])
    print(len(hic_list))
    return hic_list["Hi-C Accession"].to_list()
    
def download_encode(output_dir, url):
    root_path = os.getcwd()
    os.chdir(output_dir)
    os.system(f"wget {url}")
    os.chdir(root_path)

def download_hic(dataset_list, output_dir):
    for dataset in dataset_list:
        if "EN" in dataset:
            url = f"https://www.encodeproject.org/files/{dataset}/@@download/{dataset}.hic"
            download_encode(output_dir, url)
        else:
            print(f"Unknown dataset {dataset}")

def main():
    parser = argparse.ArgumentParser(description="Download Hi-C datasets based on a provided CSV file.")
    
    parser.add_argument("--hiclist_path", type=str, required=True, help="Path to the hic list file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where downloaded Hi-C files will be saved.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset_list = parse_hiclist(args.hiclist_path)
    download_hic(dataset_list, args.output_dir)

if __name__ == "__main__":
    main()
