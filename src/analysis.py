import pandas as pd

# Load the reference file (classification for each sample)
ref = pd.read_csv("../data/pcaa-master-platinum_set_2020.tsv", sep="\t", header=0, usecols=[0,7,8,12,13,28,32])

# Only keep TCGA samples
ref = ref[ref['study'] == 'TCGA']

# Load the oncoprint matrix after querying genes against all TCGA studies
res = pd.read_csv("../data/PATIENT_DATA_oncoprint.tsv", sep="\t", header=0, skiprows=[1,2,3,4])

# Filter samples out - only keep samples/patients with ecDNA classification
res = res[['track_name'] + list(res.columns[res.columns.isin(ref['patient_barcode'])])]

## Group the columns into ecDNA+ vs ecDNA-
# Strategy 1: ecDNA- = Non-circular; ecDNA+ = otherwise
ecdna_neg = ref[ref['sample_classification'] == 'Non-circular']['patient_barcode']
ecdna_pos = ref[ref['sample_classification'] != 'Non-circular']['patient_barcode']
pos1 = list(res.columns[res.columns.isin(ecdna_pos)])
print("Oncoprint1: {}, {}".format(pos1[-1], len(pos1)))
cols_met1 = ['track_name'] + pos1 + list(res.columns[res.columns.isin(ecdna_neg)])
df1 = res[cols_met1]
df1.to_csv("../data/oncoprint_1.tsv", sep="\t", index=False)

# Strategy 2: ecDNA+ = Circular; ecDNA- = otherwise
ecdna_pos = ref[ref['sample_classification'] == 'Circular']['patient_barcode']
ecdna_neg = ref[ref['sample_classification'] != 'Circular']['patient_barcode']
pos2 = list(res.columns[res.columns.isin(ecdna_pos)])
print("Oncoprint2: {}, {}".format(pos2[-1], len(pos2)))
cols_met2 = ['track_name'] + pos2 + list(res.columns[res.columns.isin(ecdna_neg)])
df2 = res[cols_met1]
df2.to_csv("../data/oncoprint_2.tsv", sep="\t", index=False)