import argparse
import numpy as np
import pandas as pd
import networkx as nx


def abs_max(row_x):
    row_max = np.max(row_x)
    row_min = np.min(row_x)
    if row_max > abs(row_min):
        return row_max
    return row_min


def select_edges(net_df: pd.DataFrame, wt_attr_name: str = 'wt',
                 max_edges: int = None):
    if max_edges is None or max_edges >= net_df.shape[0]:
        return net_df
    return net_df.nlargest(n=max_edges, columns=wt_attr_name)


def main(network_file, output_file, min_edges = 100000,
         max_edges = 2000000, num_steps = 10000, wt_attr='wt'):
    print("Network File : ", network_file)
    print("Output File  : ", output_file)
    print("Min. Edges   : ", min_edges)
    print("Max. Edges   : ", max_edges)
    print("Num. Steps   : ", num_steps)
    print("Wt. Attr     : ", wt_attr)
    net_df = pd.read_csv(network_file, sep="\t" )
    net_df = net_df.loc[:, ['edge', wt_attr]]
    net_df[['source', 'target']] = net_df['edge'].str.split(pat='-',expand=True)
    net_df = net_df.loc[:, ['source', 'target', wt_attr]]
    print(net_df.columns)
    net_df.sort_values(by=wt_attr, inplace=True, ascending=False)
    df_edges = max_edges - min_edges
    #th_lst = [float(df_edges*x)/ndiv for x in range(0, ndiv+1)]
    th_lst = [x for x in range(min_edges, max_edges, num_steps)]
    edge_ct = [0 for _ in range(len(th_lst))]
    node_ct = [0 for _ in range(len(th_lst))]
    density_ct = [0.0 for _ in range(len(th_lst))]
    minwt_ct = [0.0 for _ in range(len(th_lst))]
    maxwt_ct = [0.0 for _ in range(len(th_lst))]
    for i,j in enumerate(th_lst):
        x = int(j)
        sub_net_df = net_df.head(x)
        rev_net: nx.Graph = nx.from_pandas_edgelist(sub_net_df, edge_attr=wt_attr)
        node_ct[i] = nx.number_of_nodes(rev_net)
        edge_ct[i] = nx.number_of_edges(rev_net)
        density_ct[i] = nx.density(rev_net)
        minwt_ct[i] = np.min(sub_net_df[wt_attr])
        maxwt_ct[i] = np.max(sub_net_df[wt_attr])
    out_df = pd.DataFrame({"MAX_EDGES": th_lst,
                           "EDGE_COUNT": edge_ct,
                           "NODE_COUNT": node_ct,
                           "DENSITY": density_ct,
                           "MAX_WT": maxwt_ct,
                           "MIN_WT": minwt_ct})
    out_df.to_csv(output_file)


if __name__ == "__main__":
    PROG_DESC = """Compute network density vs Threshold"""
    ARGPARSER = argparse.ArgumentParser(description=PROG_DESC)
    ARGPARSER.add_argument("network_file")
    ARGPARSER.add_argument("output_file")
    ARGPARSER.add_argument("-n", "--min_edges", type=int, default=100000,
                           help="""minimum eges in the network""")
    ARGPARSER.add_argument("-x", "--max_edges", type=int, default=2000000,
                           help="""maximum eges in the network""")
    ARGPARSER.add_argument("-s", "--num_steps", type=int, default=10000,
                           help="""minimum eges in the network""")
    ARGPARSER.add_argument("-t", "--wt_attr", type=str, default='wt',
                           help="name of weight attribute")
    CMDARGS = ARGPARSER.parse_args()
    main(CMDARGS.network_file, CMDARGS.output_file, CMDARGS.min_edges,
         CMDARGS.max_edges, CMDARGS.num_steps, CMDARGS.wt_attr)
