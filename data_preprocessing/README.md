# Reproduce the preprocessing data

Place the companion Zenodo folder at `Puget/data/`, then run all commands from
`Puget/data_preprocessing/`. Generated files go under `Puget/outputs/`.

```bash
conda activate Puget
cd Puget/data_preprocessing
```

## 1. Download datasets

```bash
wget -O ../data/gencode.v29.annotation.gtf.gz https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_29/gencode.v29.annotation.gtf.gz
gunzip ../data/gencode.v29.annotation.gtf.gz

wget -O ../data/k50.Umap.MultiTrackMappability.bw https://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/k50.Umap.MultiTrackMappability.bw

wget -O ../data/hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip ../data/hg38.fa.gz

wget -O ../data/model_fold_1.safetensors https://huggingface.co/gtca/alphagenome_pytorch/resolve/main/model_fold_1.safetensors
```

Download the 16 RNA-seq and Hi-C accessions listed in `../data/deeply_profiled.csv`:

```bash
python download_rna_list.py --rnalist_path ../data/deeply_profiled.csv --output_dir ../data/rna

python download_hic_list.py --hiclist_path ../data/deeply_profiled.csv --output_dir ../data/hic
```

## 2. Build the filtered gene table and model splits

```bash
python prepare_gene_table.py --epinformer-csv ../data/EPInformer_18377genes_combined.csv --gencode ../data/gencode.v29.annotation.gtf --chrom-sizes ../data/hg38.chrom.sizes --mappability-bw ../data/k50.Umap.MultiTrackMappability.bw --biosample-csv ../data/deeply_profiled.csv --rna-dir ../data/rna --output-csv ../outputs/filtered_genes.csv

python create_enformer_split_data.py --annotation-csv ../outputs/filtered_genes.csv --biosample-csv ../data/deeply_profiled.csv --native-bed ../data/enformer_human_split/sequences.bed --chrom-sizes ../data/hg38.chrom.sizes --output-dir ../outputs/enformer_data

python create_borzoi_split_data.py --annotation-csv ../outputs/filtered_genes.csv --biosample-csv ../data/deeply_profiled.csv --native-bed ../data/borzoi_human_split/sequences_human.bed --chrom-sizes ../data/hg38.chrom.sizes --output-dir ../outputs/borzoi_data
```

The scripts write ordered gene manifests, 196,608 bp Enformer or 524,288 bp
Borzoi BED/BEDPE files, and `traincell_*gene.npy` / `testcell_*gene.npy` labels.
The expected split counts are Enformer **14,274 / 1,439 / 1,707** and Borzoi
**13,768 / 1,873 / 1,690** for train / valid / test. The labels are
`log2(TPM + 1)` with biosamples as rows and genes in BED order as columns.

## 3. Generate 1 kb sequence embeddings

Run one command per model. Each command processes the train, valid, and test
BED files. Enformer and Borzoi fetch their pretrained weights; AlphaGenome uses
the fold-1 checkpoint downloaded in Section 1.

```bash
conda activate enformer-pytorch
python embed_enformer_196k_post_pointwise_1kb.py --gpu-ids 0 --bed-dir ../outputs/enformer_data --fasta ../data/hg38.fa --out-root ../outputs/seq_embeddings
```

```bash
conda activate borzoi-pytorch
python embed_borzoi_524k_trunk_1kb.py --gpu-ids 0 --bed-dir ../outputs/borzoi_data --fasta ../data/hg38.fa --out-root ../outputs/seq_embeddings
```

```bash
conda activate alphagenome-pytorch
python embed_alphagenome_524k_post_embedder_1kb.py --gpu-ids 0 --bed-dir ../outputs/borzoi_data --fasta ../data/hg38.fa --checkpoint ../data/model_fold_1.safetensors --out-root ../outputs/seq_embeddings
```

Pass `--gpu-ids 0 1` to shard a model across two GPUs. Each command writes an
array and metadata for each split under `../outputs/seq_embeddings/`. Existing
arrays are protected; use `--overwrite` only to regenerate them.

## 4. Convert Hi-C and extract contact windows

The conversion commands turn all 16 `.hic` files into chromosome-wide sparse
pickles at 1 kb. The HiCFoundation baseline uses observed contacts with `NONE`
normalization (RAW counts), matching the contact representation used for its
pretraining; it also needs the accession-wide raw `total_count` for its count
token. Puget uses observed/expected contacts with `SCALE` normalization (O/E).
The extraction commands create matching train/valid/test gene windows and
rebins them to 1,024 bp: 192 × 192 for Enformer, 512 × 512 for Borzoi.
These files are large, so allow substantial disk space in `../outputs/`.

```bash
conda activate Puget

for hic in ../data/hic/*.hic; do
    accession="${hic##*/}"
    output="../outputs/hic_raw_pkls/${accession%.hic}.pkl"
    if [ -s "$output" ]; then
        echo "[SKIP] $output"
    else
        python hic2array_raw.py --input-hic "$hic" --output-pkl "$output" || break
    fi
done
```

```bash
for hic in ../data/hic/*.hic; do
    accession="${hic##*/}"
    output="../outputs/hic_oe_pkls/${accession%.hic}.pkl"
    if [ -s "$output" ]; then
        echo "[SKIP] $output"
    else
        python hic2array_oe.py --input-hic "$hic" --output-pkl "$output" || break
    fi
done
```

```bash
python extract_hicfoundation_raw_windows.py --hic-pkl-dir ../outputs/hic_raw_pkls --output-root ../outputs/hic_windows/hicfoundation_enformer --window-bp 196608 --bedpe train=../outputs/enformer_data/Enformer_genes_196k_196k_train.bedpe --bedpe valid=../outputs/enformer_data/Enformer_genes_196k_196k_valid.bedpe --bedpe test=../outputs/enformer_data/Enformer_genes_196k_196k_test.bedpe
```

```bash
python extract_hicfoundation_raw_windows.py --hic-pkl-dir ../outputs/hic_raw_pkls --output-root ../outputs/hic_windows/hicfoundation_borzoi --window-bp 524288 --bedpe train=../outputs/borzoi_data/Borzoi_genes_524k_524k_train.bedpe --bedpe valid=../outputs/borzoi_data/Borzoi_genes_524k_524k_valid.bedpe --bedpe test=../outputs/borzoi_data/Borzoi_genes_524k_524k_test.bedpe
```

```bash
python extract_puget_oe_windows.py --hic-pkl-dir ../outputs/hic_oe_pkls --output-root ../outputs/hic_windows/puget_enformer --window-bp 196608 --bedpe train=../outputs/enformer_data/Enformer_genes_196k_196k_train.bedpe --bedpe valid=../outputs/enformer_data/Enformer_genes_196k_196k_valid.bedpe --bedpe test=../outputs/enformer_data/Enformer_genes_196k_196k_test.bedpe
```

```bash
python extract_puget_oe_windows.py --hic-pkl-dir ../outputs/hic_oe_pkls --output-root ../outputs/hic_windows/puget_borzoi --window-bp 524288 --bedpe train=../outputs/borzoi_data/Borzoi_genes_524k_524k_train.bedpe --bedpe valid=../outputs/borzoi_data/Borzoi_genes_524k_524k_valid.bedpe --bedpe test=../outputs/borzoi_data/Borzoi_genes_524k_524k_test.bedpe
```

Each window directory contains `train/`, `valid/`, and `test/`, with one
`<Hi-C accession>.pkl` per split. The conversion loops skip existing nonempty
pickles; the window extractors process accession pickles sequentially and skip
existing output files. Each extractor defaults to one numerical thread per
process.
