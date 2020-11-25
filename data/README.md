
yeast networks from simulated datasets
========================

v1:
---

GRNBoost, ARACNE, TINGe have partial networks.
   CLR, mrnet and pearson have complete networks. 
   pearson values as absolute ranging from 0 to 1.

v2:
---

   ARACNE, TINGe have complete networks. GRNBoost has only 500K edges.
   pearson values can be either positive or negative ranging from -1 to +1

v3:
---

   v2 with WGCNA added since for real networks, we dont have pcc



arabidopsis networks from real datasets
========================

arabidopsis-edges-final-test-v1.csv.gz
---------------------------------------

   Arabidopsis networks with clr, grnboost, aracne, tinge, aracne, mrnet, wgcna.
   Missing weights are empty. This will be the final weights
