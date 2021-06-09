import argparse
import pandas as pd
import numpy as np

def select_edges_ct(net_df: pd.DataFrame, wt_attr_name: str = 'wt',
                    max_edges: int = None,
                    reverse_order: bool = False):
    if max_edges is None or max_edges >= net_df.shape[0]:
        return net_df
    cur_cols = net_df.columns
    maxwt_attr_name = wt_attr_name + '_max'
    net_df = net_df.assign(maxwt_col=net_df[wt_attr_name].abs().values)
    print(net_df.columns)
    if reverse_order is True:
        net_df = net_df.nsmallest(n=max_edges, columns='maxwt_col')
    else:
        net_df = net_df.nlargest(n=max_edges, columns='maxwt_col')
    return net_df.loc[:, cur_cols]


def select_edges_wt(net_df: pd.DataFrame, wt_attr_name: str = 'wt',
                    th_weight: float = None, reverse_order: bool = False):
    if reverse_order is True:
        if th_weight is None or th_weight > np.min(net_df[wt_attr_name]):
            return net_df
        return net_df.loc[net_df[wt_attr_name] < th_weight]
    else:
        print("Min value : ", th_weight, np.min(net_df[wt_attr_name]))
        if th_weight is None or th_weight < np.min(net_df[wt_attr_name]):
            return net_df
        return net_df.loc[net_df[wt_attr_name] > th_weight]

def main(network_file, output_file, wt_attr, th_weight, max_edges, reverse_order):
    print(network_file, output_file, wt_attr, th_weight, max_edges, reverse_order)
    net_df = pd.read_csv(network_file, sep='\t', header=0)
    print("Loaded network file : ", network_file)
    print(net_df.columns, net_df.dtypes, th_weight, max_edges)
    if th_weight is not None:
        tmp_df = select_edges_wt(net_df, wt_attr_name=wt_attr,
                                 th_weight=th_weight, reverse_order=reverse_order)
    else:
        tmp_df = net_df
    if max_edges is not None:
        out_df = select_edges_ct(tmp_df, wt_attr_name=wt_attr,
                                 max_edges=max_edges, reverse_order=reverse_order)
    else:
        out_df = tmp_df
    out_df.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    PROG_DESC = """Compute network with edge weights above given Threshold"""
    ARGPARSER = argparse.ArgumentParser(description=PROG_DESC)
    ARGPARSER.add_argument("network_file")
    ARGPARSER.add_argument("-w", "--th_weight", type=float, default=0.0,
                           help="""weight threshold for the network.
                           default is lower bound, upper bound if -r is given""")
    ARGPARSER.add_argument("-x", "--max_edges", type=int,
                           help="""maximum eges in the network""")
    ARGPARSER.add_argument("-t", "--wt_attr", type=str, default='wt',
                           help="name of weight attribute")
    ARGPARSER.add_argument("-r", "--reverse_order", action='store_true',
                           help="""Order the edges ascending order""")
    ARGPARSER.add_argument("output_file")
    CMDARGS = ARGPARSER.parse_args()
    main(CMDARGS.network_file, CMDARGS.output_file,
         CMDARGS.wt_attr, CMDARGS.th_weight, CMDARGS.max_edges,
         CMDARGS.reverse_order)
