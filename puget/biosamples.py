import csv

# GM12878 and K562 are held out; their labels follow the 14 training rows.
HELD_OUT_BIOSAMPLES = ("GM12878", "K562")


def load_biosample_table(path):
    with open(path, newline="") as handle:
        entries = [(row["Biosample"].strip(), row["Hi-C accession"].strip())
                   for row in csv.DictReader(handle)]
    by_name = dict(entries)
    missing = [name for name in HELD_OUT_BIOSAMPLES if name not in by_name]
    if missing or len(by_name) != len(entries):
        raise ValueError(f"{path}: duplicate biosamples or missing held-out {missing}")
    names = [name for name, _ in entries if name not in HELD_OUT_BIOSAMPLES]
    names.extend(HELD_OUT_BIOSAMPLES)
    return [(index, by_name[name], name) for index, name in enumerate(names)]
