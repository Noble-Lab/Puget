# The scripts are adapted from https://github.com/Noble-Lab/HiCFoundation_paper/blob/main/notebooks/pretrain_data.ipynb
import pandas as pd
import os
import argparse

def parse_rnalist(rnalist_path):
    rna_list = pd.read_csv(rnalist_path, names = ["RNA Accession"])
    print(len(rna_list))
    return rna_list["RNA Accession"].to_list()
   
def download_encode(output_dir, url):
    root_path = os.getcwd()
    os.chdir(output_dir)
    os.system(f"wget {url}")
    os.chdir(root_path)

def download_rna(dataset_list, output_dir):
    for dataset in dataset_list:
        if "EN" in dataset:
            url = f"https://www.encodeproject.org/files/{dataset}/@@download/{dataset}.tsv"
            download_encode(output_dir, url)
        else:
            print(f"Unknown dataset {dataset}")

def main():
    parser = argparse.ArgumentParser(description="Download RNA datasets based on provided CSV files.")
    parser.add_argument("--rnalist_path", type=str, required=True, help="Path to the rna list file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where downloaded RNA-seq files will be saved.")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset_list = parse_rnalist(args.rnalist_path)
    download_rna(dataset_list, args.output_dir)

if __name__ == "__main__":
    main()
