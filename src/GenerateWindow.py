#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os,re
import time
import pandas as pd
import numpy as np

minda = '/NAS/wg_zql/SoftWare/minda-main/minda.py'

## Tools Res
def minda_vcf(input_vcfs,out_dir):
    ### Running minda 
    Toolsvcf = ' '.join(input_vcfs)
    os.system('{Minda} ensemble --vcfs {Toolsvcf} --out_dir {OUT} --min_support 1 --multimatch'.format(
        Minda=minda,Toolsvcf=Toolsvcf,OUT=out_dir))
    
#####

def parse_mindaout(SampleID,out_dir,Len_cutoff = 10000):
    f = os.path.join(out_dir, 'None_support.tsv')
    Record = pd.read_csv(f,sep='\t')
    ###
    vcf_df = os.path.join(out_dir, 'None_minda_ensemble.vcf')
    with open(vcf_df,'r') as fin:
        records = [x.strip().split("\t") for x in fin.readlines() if not re.search('##', x)]
    ## generate dataframe
    vcfDf = pd.DataFrame.from_records(records[1:])
    vcfDf.columns = records[0]
    vcfDf['POS'] = vcfDf['POS'].astype('int64')
    vcfDf = vcfDf[['#CHROM', 'POS','ID']]
    ###
    merged_df = pd.merge(Record, vcfDf, left_on=['#CHROM_x', 'POS_x'], right_on=['#CHROM', 'POS'], how='outer')
    ### filter
    merged_df = merged_df.loc[merged_df['SVTYPE'].isin(['INS','DUP','DEL'])]
    merged_df['start'] = merged_df.apply(lambda row: min(int(row['POS_x']),int(row['POS_y'])), axis=1)
    merged_df['end'] = merged_df.apply(lambda row: max(int(row['POS_x']),int(row['POS_y'])), axis=1)
    ###
    merged_df[['#CHROM_x','start','end','SVTYPE','SVLEN','ID']].to_csv(
        os.path.join(out_dir,'%s.minda.Somatic.txt' % (SampleID)),
        sep='\t',index=None,header=None)
    ### Filter Len
    vcfDf_SSV = merged_df[merged_df['SVLEN']<=Len_cutoff]
    vcfDf_SSV[['#CHROM_x','start','end','SVTYPE','SVLEN','ID']].to_csv(
        os.path.join(out_dir,'%s.minda.Somatic.Filter10k.txt' % (SampleID)),
        sep='\t',index=None,header=None)
    return('{} Finished'.format(SampleID))


def main():
    parser = argparse.ArgumentParser(description="Process VCF files")
    parser.add_argument('-o', '--output_dir', required=True, help="Output directory")
    parser.add_argument('-s', '--sample', required=True, help="Sample ID")
    parser.add_argument('input_files', nargs='+', help="Input VCF files")
    args = parser.parse_args()
    # minda running
    minda_vcf(args.input_files,args.output_dir)
    # Filter 
    parse_mindaout(args.sample,args.output_dir)




if __name__ == "__main__":
    main()

