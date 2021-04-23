import pandas as pd

with open("tc.txt") as tf:
    tcxs = [x.strip() for x in tf.readlines()]

tcx_files = {x:"./tp_found/"+x+"_tps.txt" for x in tcxs}

def get_lines(fname):
    with open(fname) as fx:
        return [x.strip() for x in fx.readlines()]

tp_edges = {x: get_lines(fn)[1:] for x,fn in tcx_files.items()}
df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tp_edges.items() ]))
df.to_csv("./tp_found/all_tissue_edges.tsv", sep="\t", index=False)

tp_nodes = {x:list(dict.fromkeys([item for sublist in  [z.split('-') for z in y] for item in sublist])) for x,y in tp_edges.items()}
gdf = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tp_nodes.items() ]))
gdf.to_csv("./tp_found/all_tissue_nodes.tsv", sep="\t", index=False)
