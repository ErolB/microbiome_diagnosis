import csv
import pandas as pd

def process_rats(table, metadata, out_dir):
    ''' Puts data from rat study into standard format'''
    with open(metadata, 'r') as meta_file:
        sample_data = list(csv.DictReader(meta_file, delimiter=','))
    hf_samples = [item['sample'] for item in sample_data if (item['diet']=='HF') and (item['organ']=='C')]
    lc_samples = [item['sample'] for item in sample_data if (item['diet']=='LC') and (item['organ']=='C')]
    with open(table, 'r') as table_file:
        otu_data = pd.read_table(table_file, delimiter=',', index_col='KO_ID')
        hf_samples = [item for item in hf_samples if item in otu_data.columns]
        lc_samples = [item for item in lc_samples if item in otu_data.columns]
        #formatted_data = formatted_data.transpose()
        pos_data = otu_data[hf_samples]
        neg_data = otu_data[lc_samples]
    pos_data.to_csv(out_dir+'/pos_data.csv')
    neg_data.to_csv(out_dir+'/neg_data.csv')

def load_data(in_dir):
    ''' Loads data in standard format '''
    pos_data = pd.read_csv(in_dir+'/pos_data.csv', index_col='OTU_ID')
    neg_data = pd.read_csv(in_dir+'/neg_data.csv', index_col='OTU_ID')
    return pos_data, neg_data

if __name__ == '__main__':
    process_rats('metagenome_table.csv', 'clustered/metadata.csv', 'formatted_data/rats_colon_metagenome')