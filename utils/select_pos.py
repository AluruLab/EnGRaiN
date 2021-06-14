import argparse
import pandas as pd
import numpy as np


tissue_types = ['flower', 'leaf', 'root', 'rosette', 'seed', 'seedling1wk',
                'seedling2wk', 'shoot', 'wholeplant']

condition_types = ['chemical', 'development', 'hormone-aba-iaa-ga-br',
                   'hormone-ja-sa-ethylene','light','nutrients',
                   'stress-light', 'stress-other', 'stress-pathogen', 
                   'stress-salt-drought', 'stress-temperature']
#tissue_types = ['flower']
#condition_types = ['chemical']

def network_stats(pos_dir, min_wt_file, out_dir):
    net_max_stats = []
    dfx = pd.read_csv(min_wt_file, sep="\t")
    for tx in tissue_types + condition_types:
        pos_file = pos_dir +  "/" + tx + "-positives.csv"
        pos_df = pd.read_csv(pos_file)
        print("Loaded pos file : ", pos_file, pos_df.shape)
        clr_min = dfx.loc[(dfx.TISSUE == tx) & (dfx.WT_ATTR == 'clr')].MIN_WT.iloc[0]
        wgcna_min = dfx.loc[(dfx.TISSUE == tx) & (dfx.WT_ATTR == 'wgcna')].MIN_WT.iloc[0]
        aracne_min = dfx.loc[(dfx.TISSUE == tx) & (dfx.WT_ATTR == 'aracne')].MIN_WT.iloc[0]
        mrnet_min = dfx.loc[(dfx.TISSUE == tx) & (dfx.WT_ATTR == 'mrnet')].MIN_WT.iloc[0]
        tinge_min = dfx.loc[(dfx.TISSUE == tx) & (dfx.WT_ATTR == 'tinge')].MIN_WT.iloc[0]
        grnb_min = dfx.loc[(dfx.TISSUE == tx) & (dfx.WT_ATTR == 'grnboost')].MIN_WT.iloc[0]
        pos_df.fillna(0.0)
        pos_df = pos_df.loc[ (pos_df.clr > clr_min) | (pos_df.wgcna > wgcna_min) |
                   (pos_df.aracne > aracne_min) | (pos_df.mrnet > mrnet_min) |
                   (pos_df.tinge > tinge_min) | (pos_df.grnboost > grnb_min) ]
        pos_out_file = out_dir +  "/" + tx + "-positives.csv"
        print("Writing to pos file : ", pos_out_file, pos_df.shape)
        pos_df.to_csv(pos_out_file, index=False)
        neg_file = pos_dir +  "/" + tx + "-negatives.csv"
        neg_df = pd.read_csv(neg_file)
        print("Loaded neg file : ", neg_file, neg_df.shape)
        neg_df = neg_df.loc[ (neg_df.clr < clr_min) | (neg_df.wgcna < wgcna_min) |
                   (neg_df.aracne < aracne_min) | (neg_df.mrnet < mrnet_min) |
                   (neg_df.tinge < tinge_min) | (neg_df.grnboost < grnb_min) ]
        neg_out_file = out_dir +  "/" + tx + "-negatives.csv"
        neg_df.to_csv(neg_out_file, index=False)
        print("Writing to neg file : ", neg_out_file, neg_df.shape)


if __name__ == "__main__":
    PROG_DESC = """Find Sensitivity Statistics"""
    ARGPARSER = argparse.ArgumentParser(description=PROG_DESC)
    ARGPARSER.add_argument("pos_dir")
    ARGPARSER.add_argument("min_wt_file")
    ARGPARSER.add_argument("out_dir")
    CMDARGS = ARGPARSER.parse_args()
    print(CMDARGS.pos_dir, CMDARGS.min_wt_file, CMDARGS.out_dir)
    network_stats(CMDARGS.pos_dir, CMDARGS.min_wt_file, CMDARGS.out_dir)
 
