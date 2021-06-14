1. (Current Bioinformatics, 2010) : A review of ensemble methods in bioinformatics. Paper: https://www.maths.usyd.edu.au/u/pengyi/publication/EnsembleBioinformatics-v6.pdf. This is a general introduction to ensemble methods, does not discuss GRNs. From the paper :

       ... we describe the application of ensemble methods in bioinformatics in three broad topics.They are as follow: 
       Classification of gene expression microarray data and MS-based proteomics data;
       Gene-gene interaction identification using single nucleotide polymorphism (SNPs) data from G-WA studies;
       Prediction of regulatory elements from DNA and protein sequences.

2. (Bioinformatics, 2010) : Revealing differences in gene network inference algorithms on the network level by ensemble methods. Paper: https://doi.org/10.1093/bioinformatics/btq259
	- Ensemble of datasets
	- generate E = 300 different datasets for sample size 200 and another E = 300 datasets for sample size 20 using the same subnetwork : this is what is called ensemble of datasets
	- Optimal cutoff value for each dataset, Di, used to declare edges significant is obtained by maximizing the F-score,
	- This results in E different F-scores, correspondingly E inferred networks
	- The rest of the paper analyses the newtork properties : histogram etc of the 600 x 4 (methods) networks.

3. (PLOS) : Ensemble methods for stochastic networks with special reference to the biological clock of Neurospora crass. URL: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196435
   - This is stochastic clock network, not gene network.

4. Netbenchmark: https://link.springer.com/article/10.1186/s12859-015-0728-4 

5. (IEEE SSCI 2015): Study of Normalization and Aggregation Approaches for Consensus Network Estimation. URL: https://ieeexplore.ieee.org/abstract/document/7376731
    - Methods: RankSum, TopKNet,
    - Methods: Normalize and aggregate: 
    - Normalization methods:
      - Identity : no change
      - Ranking,
      - Scaling with mean and sd,
      - SumL: Ranking as an average of relative ranks of the two genes
      - ScaleL : Scaling as an average of relative scalings of the two genes involved in the interaction
    - Aggregation methods:
      - Sum
      - Top 3 : eTop3 is the 3rd rank order filter of the N values, which is defined as taking the 3rd greatest value of the N values of each link.
      - Median
    - 5 x 3 = 15 different combinations
	- Homogeneous scenario reflects the case where the individual networks have been inferred with different algorithms but with the same type of data like gene expression	
	- Heterogeneous scenario reflects the case where the individual networks have been inferred from very different kind of data
	- Note on methodology:  "...use the Area Under Precision Recall (AUPR) but we do not evaluate the whole network and only take into account the most confident edges... In order to have statistically significant measures, we choose to evaluate the 20% highest confidence edges as is done in[8], obtaining the AUPR20 measure..."
	- Conclustions: "... We use five different TrueNet networks to evaluate the consensus algorithms, that are generated with three different GRN simulators: GNW simulator [15], SynTReN simulator [16] and Rogers... In the homogeneous case, it can be concluded that consensus network algorithms allow to improve the inference. this case almost shows no significant differences between methods, we think that the best option is to use the RankSum or IdSum...In the Heterogeneous scenario, we observe large differences between different consensus proposals, where some of them reach worst results compared to the average individual network. We can confirm that the IdSum method that was used in [11] in a Heterogeneous scenario is a good choice for this case. But, using the ScaleL as normalization step and Sum as aggregation step provides even better results"

7. (GRN Book chapter 2018; most relevant) : Unsupervised GRN Ensemble. Paper: https://link.springer.com/protocol/10.1007/978-1-4939-8882-2_12. Schihub link : https://sci-hub.se/10.1007/978-1-4939-8882-2_12#
	- Sort of a Journal version of the SSCI conf. paper.
	- Current State of the art : IdSum, RankSum, RankMedian
	- Same methods as the Conf. paper
	- Also uses NetBenchMark datasets
	- Also uses DREAM5 data : E.coli network
	- Also D. melanogaster dataset from a paper which constructs gene network from multiple different datasets : chip-seq, expression datasets
	- Summary: ScaleLSum is recommended for Homogeneous, IdSum is recommended for Heterogeneous.
	- Also discuss pairwise aggregation but no experiments are done.
	
8. (BMC Bioinformatics 2017): Study of Meta-analysis strategies for network inference using information-theoretic approaches. URL: https://link.springer.com/article/10.1186/s13040-017-0136-6 .
	- Only discusses the combination of MI methods
	- Includes three different way of combing: merging of datasets, ensemble of networks, and aggregation of MI matrices .
	- For datasets: Use of COMBAT for normalization
	- For networks: 
		 - RankSum
		 - Internal Quality Control Index : based on Metaqc: https://doi.org/10.1093%2Fnar%2Fgkr1071
		 - Median confidence score
	- Similar methods for aggregating correlation matrices
	- Presents only as a framework where a user can experiment with different options above
	
9. Other tangentailly related papers :
	- Network inference performance complexity: a consequence of topological, experimental and algorithmic determinants. URL: https://academic.oup.com/bioinformatics/article/35/18/3421/5319942?login=true .
	- Stability in GRN Inference. URL: https://link.springer.com/protocol/10.1007/978-1-4939-8882-2_14
