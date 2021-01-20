import argparse
import pandas as pd
import numpy as np

def net_stats(net_df, n_edges, pos_df, neg_df, wt_attr):
    top_df = net_df.head(n_edges)
    tp_shape = top_df[ top_df.edge.isin(pos_df.edge) ].shape
    fp_shape = top_df[ top_df.edge.isin(neg_df.edge) ].shape
    max_wt = np.max(top_df[wt_attr])
    min_wt = np.min(top_df[wt_attr])
    ntp = float(tp_shape[0])
    nfp = float(fp_shape[0])
    nfn = float(pos_df.shape[0]) - ntp
    ntn = float(neg_df.shape[0]) - nfp
    stats_dct = {
            'Edges': n_edges, 'MaxWt': max_wt, 'MinWt': min_wt,
            'TP': ntp, 'FP': nfp, 'TN': ntn, 'FN': nfn,
            'Sensitivity/Recall': ntp/(ntp+nfn),
            'Specificity': ntn/(ntn+nfp) if (ntn+nfp) > 0 else 1.0,
            'Precision': ntp/(ntp+nfp) if (ntp+nfp) > 0 else 1.0,
            'Accuracy': (ntp+ntn)/(ntp+ntn+nfp+nfn),
            'F1' : (2.0*ntp)/((2*ntp) + nfp + nfn)
            }
    return stats_dct


def network_stats_df(net_df, pos_df, neg_df, wt_attr):
    stats_lst = [net_stats(net_df, nx, pos_df, neg_df, wt_attr) for nx in range(100_000, 10_000_000, 100_000)]
    return pd.DataFrame(data=stats_lst)


def network_stats(network_file, pos_file, neg_file, wt_attr):
    net_df = pd.read_csv(network_file, sep="\t")
    net_df.sort_values(by=wt_attr, inplace=True, ascending=False)
    pos_df = pd.read_csv(pos_file)
    neg_df = pd.read_csv(neg_file)
    return network_stats_df(net_df, pos_df, neg_df, wt_attr)

if __name__ == "__main__":
    PROG_DESC = """Find Sensitivity Statistics"""
    ARGPARSER = argparse.ArgumentParser(description=PROG_DESC)
    ARGPARSER.add_argument("network_file")
    ARGPARSER.add_argument("pos_file")
    ARGPARSER.add_argument("neg_file")
    ARGPARSER.add_argument("output_file")
    ARGPARSER.add_argument("-t", "--wt_attr", type=str, default='wt',
                           help="name of weight attribute")
    CMDARGS = ARGPARSER.parse_args()
    print(CMDARGS.network_file, CMDARGS.pos_file, CMDARGS.neg_file,
          CMDARGS.output_file, CMDARGS.wt_attr)
    odf = network_stats(CMDARGS.network_file, CMDARGS.pos_file,
                        CMDARGS.neg_file, CMDARGS.wt_attr)
    odf.to_csv(CMDARGS.output_file, sep="\t", index=False)
 
