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

def network_stats(preds_dir, ntop = 4000000):
    net_max_stats = []
    for tx in tissue_types + condition_types:
        network_file = preds_dir +  "/" + tx + "-iqr-preds.tsv"
        print("Loading network file : ", network_file)
        net_df = pd.read_csv(network_file, sep="\t")
        for wt_attr in net_df.columns:
            if wt_attr == "edge":
                continue
            wt_col = net_df.loc[net_df[wt_attr] >0, wt_attr]
            ncount = len(wt_col)
            min_wt = 0.0
            if ncount > 0:
               min_wt = wt_col.nlargest(ntop).tail(1).iloc[0]
            net_mwt = {'TISSUE' : tx, 'NCOUNT': ncount,
                    'WT_ATTR': wt_attr, 'MIN_WT' : min_wt}
            print(net_mwt)
            net_max_stats.append(net_mwt)
        print("Completed network file : ", network_file)
    return pd.DataFrame(data=net_max_stats)

if __name__ == "__main__":
    PROG_DESC = """Find Sensitivity Statistics"""
    ARGPARSER = argparse.ArgumentParser(description=PROG_DESC)
    ARGPARSER.add_argument("preds_dir")
    ARGPARSER.add_argument("output_file")
    CMDARGS = ARGPARSER.parse_args()
    print(CMDARGS.preds_dir, CMDARGS.output_file)
    odf = network_stats(CMDARGS.preds_dir)
    odf.to_csv(CMDARGS.output_file, sep="\t", index=False)
 
