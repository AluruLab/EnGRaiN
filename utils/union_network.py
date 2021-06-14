import argparse
import numpy as np
import pandas as pd
import datetime
from typing import List

def load_reveng_network(net_file, wt_attr_name, mx_wt, all_columns):
    if net_file.endswith("csv"):
       tmp_df = pd.read_csv(net_file)
    else:
       tmp_df = pd.read_csv(net_file, sep="\t")
    if mx_wt is not None:
        tmp_df = tmp_df.loc[tmp_df['wt'] > mx_wt, ]
    if all_columns is True:
        tmp_df = tmp_df.sort_values(by='clr', # TODO: change this?
               ascending=False).drop_duplicates(subset=['edge'], keep='first')
        col_renames = {x : wt_attr_name + "-" + x for x in tmp_df.columns if x != 'edge'}
        tmp_df = tmp_df.rename(columns=col_renames)
    else:
        tmp_df = tmp_df.rename(columns={'wt': wt_attr_name})
        tmp_df = tmp_df.loc[:, ['edge', wt_attr_name]]
        tmp_df = tmp_df.sort_values(by=wt_attr_name,
               ascending=False).drop_duplicates(subset=['edge'], keep='first')
    return tmp_df


def combine_network(network_files, network_names=None,
                    network_wts: List[float] = None, avg_wt=True,
                    max_wt=True, all_columns=False):
    # print(max_wt)
    if network_names is None:
        network_names = ['wt_'+str(ix) for ix in range(len(network_files))]
    cmb_network = pd.DataFrame(columns=['edge'])
    ix = 0
    netx = len(network_names)
    for nx_name, nx_file, mx_wt in zip(network_names, network_files, network_wts):
        ndf = load_reveng_network(nx_file, nx_name, mx_wt, all_columns)
        cmb_network = cmb_network.merge(ndf, how='outer', on=['edge'])
        now = datetime.datetime.now()
        ix += 1
        print(now.strftime("%Y-%m-%d %H:%M:%S"), ": Loaded :", ix, "/",
                netx, str(nx_name), nx_file, 
                ndf.shape, cmb_network.columns, cmb_network.shape)
    if avg_wt is True:
        cmb_network['avgwt'] = cmb_network[network_names].mean(axis=1)
    if max_wt is True:
        cmb_network['wt'] = cmb_network[network_names].max(axis=1)
        cmb_network = cmb_network.sort_values(by='wt', ascending=False)
        print(now.strftime("%Y-%m-%d %H:%M:%S"),  ": Finished : ",
                cmb_network.columns, cmb_network.shape)
    return cmb_network


def main(network_names, network_files, out_file,
         net_wt, avg_wt, max_wt, all_columns):
    if network_names:
        network_names = network_names.split(",")
        if net_wt is not None:
            network_wts = [float(x) for x in net_wt.split(",")]
        else:
            network_wts = [None for _ in network_names]
        if len(network_names) == len(network_files) and len(network_wts) == len(network_files):
            combine_df = combine_network(network_files, network_names,
                                     network_wts, avg_wt, max_wt, all_columns)
        else:
            print("Network names, files, weights should be of equal length",
                  len(network_files), len(network_names), len(network_wts))
            return False
        if out_file.name.endswith("csv"):
            combine_df.to_csv(out_file, index=False)
        else:
            combine_df.to_csv(out_file, sep='\t', index=False)
    return True


if __name__ == "__main__":
    PROG_DESC = """
    Finds a union of input networks.
    Network union is computed as the union of edges of the input networks.
    Outputs a tab-seperated values with weights corresponding to each network
    in a sperate column, and maximum and average weight.
    """
    PARSER = argparse.ArgumentParser(description=PROG_DESC)
    PARSER.add_argument("network_files", nargs="+",
                        help="""network build from a reverse engineering methods
                                (currenlty supported: eda, adj, tsv)""")
    PARSER.add_argument("-n", "--network_names", type=str,
                        help="""comma seperated names of the network;
                                should have as many names as the number of networks""")
    PARSER.add_argument("-x", "--max_net_wt", type=str,
                        help="""max weights of edges to output""")
    PARSER.add_argument("-g", "--no_wt_avg", action='store_false',
                        help="""compute the average wt. (default: True)""")
    PARSER.add_argument("-m", "--no_wt_max", action='store_false',
                        help="""compute the average max wt. (default: True)""")
    PARSER.add_argument("-c", "--all_columns", action='store_true',
                        help="""compute the average wt. (default: False)""")
    PARSER.add_argument("-o", "--out_file",
                        type=argparse.FileType(mode='w'), required=True,
                        help="output file in tab-seperated format")
    ARGS = PARSER.parse_args()
    print("""
       ARG : network_files  : %s
       ARG : network_names  : %s
       ARG : max_net_wt     : %s
       ARG : wt_avg         : %s
       ARG : wt_max         : %s
       ARG : all_columns    : %s
       ARG : out_file       : %s """ %
          (str(ARGS.network_files), str(ARGS.network_names),
           str(ARGS.max_net_wt), str(ARGS.no_wt_avg), str(ARGS.no_wt_max),
           str(ARGS.all_columns), str(ARGS.out_file)))
    if not main(ARGS.network_names, ARGS.network_files, ARGS.out_file,
            ARGS.max_net_wt, ARGS.no_wt_avg, ARGS.no_wt_max, ARGS.all_columns):
        PARSER.print_usage()
