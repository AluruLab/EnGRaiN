
yeast networks from simulated datasets
========================

yeast-edge-weights-v1
---

| edge | prediction | pcc | clr | aracne | grnboost | mrnet | tinge |
|------|------------|-----|-----|--------|----------|-------|-------|
| ...  | ...        | ... | ... | ...    | ...      | ...   | ...   |

GRNBoost, ARACNE, TINGe have partial networks.
CLR, mrnet and pearson have complete networks. 
pearson values as absolute ranging from 0 to 1.

yeast-edge-weights-v2
---

| edge | prediction | pcc | clr | aracne | grnboost | mrnet | tinge |
|------|------------|-----|-----|--------|----------|-------|-------|
| ...  | ...        | ... | ... | ...    | ...      | ...   | ...   |


ARACNE, TINGe have complete networks. GRNBoost has only 500K edges.
Pearson values can be either positive or negative ranging from -1 to +1.


yeast-edge-weights-v3
---

| edge | prediction | pcc | clr | aracne | grnboost | mrnet | tinge | wgcna |
|------|------------|-----|-----|--------|----------|-------|-------|-------|
| ...  | ...        | ... | ... | ...    | ...      | ...   | ...   | ...   |

Same as v2 with WGCNA added.
WGCNA replaces pcc values with : 0.5*(1 + Pearson correlation) ^ sft
where sft is generated based on scale-free topology.
For real networks, we use WGCNA values instead of PCC.


yeast-edge-weights-v4 & yeast-edge-weights-v5
---
Both similar format to v3 with different features. 
Both can be downloaded  from the followng links:

[yeast-edge-weights-v4.csv.gz](https://www.dropbox.com/s/gxv6ksnbk601wwm/yeast-edge-weights-v4.csv.gz?dl=0)
-- includes  the features clr, aracne, grnboost, mrnet, tinge, wgcna, fastggm, genenet, tigress, genie3, and inferelator.

[yeast-edge-weights-v5.csv.gz](https://www.dropbox.com/s/c7rhjs75oek1wia/yeast-edge-weights-v5.csv.gz?dl=0) 
-- includes  the features clr, aracne, grnboost, mrnet, tinge, wgcna, fastggm, genenet, tigress, genie3, inferelator, and irafnet.

arabidopsis networks from real datasets
========================

athaliana_ref_positives.csv
--------------------------------

Positive edges in true network that are found in the arabidopsis network.

athaliana_ref_negatives.csv
---------------------------------

Negative edges in true network that are found in the arabidopsis network.





athaliana_raw/ and athaliana_raw2/ folders
---
Arabidopsis networks with clr, grnboost, aracne, tinge, aracne, mrnet, wgcna.
Note that all the networks are parial networks and the entries for edges 
for which there is  no prediction are empty. 


athaliana_raw_filtered/ folders
---
Arabidopsis edges with highest confidence in networks generated by 
clr, grnboost, aracne, tinge, aracne, mrnet, wgcna.
