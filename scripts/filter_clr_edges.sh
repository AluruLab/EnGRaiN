
#usage: filter_network.py [-h] [-w TH_WEIGHT] [-x MAX_EDGES] [-t WT_ATTR] [-r]
#                         network_file output_file
#
#Compute network with edge weights above given Threshold
#
#positional arguments:
#  network_file
#  output_file
#
#optional arguments:
#  -h, --help            show this help message and exit
#  -w TH_WEIGHT, --th_weight TH_WEIGHT
#                        weight threshold for the network. default is lower
#                        bound, upper bound if -r is given
#  -x MAX_EDGES, --max_edges MAX_EDGES
#                        maximum eges in the network
#  -t WT_ATTR, --wt_attr WT_ATTR
#                        name of weight attribute
#  -r, --reverse_order   Order the edges ascending order

#for x in `cat clr_tc.csv`; do 
#    y=`printf "$x" | cut -d',' -f 1`  
#    z=`printf "$x" | cut -d',' -f 2` 
#    echo $y $z `ls clr_preds/${y}-iqr-preds.tsv`
#done
thresh=clr_preds 
mkdir -p $thresh

for x in `cat clr_tc.csv`; do 
    y=`printf "$x" | cut -d',' -f 1`  
    z=`printf "$x" | cut -d',' -f 2` 
    net_file=ensemble_preds1/${y}-iqr-preds.tsv
    of1=$thresh/ensemble_${y}_1X.tsv
    ((z--))
    echo Tissue : $y Threshold : $z Input :  `ls ensemble_preds1/${y}-iqr-preds.tsv`
    if [ ! -f $of1 ] ; then
      echo python filter_network.py -x $z -t clr $net_file  $of1
      python filter_network.py -x $z -t clr $net_file  $of1
    fi
    v=$(( 2*z ))
    # echo "Threshold : " $v 
    # echo python eval/filter_network.py  -x $v $net_file  $of2
    # python eval/filter_network.py  -x $v $net_file  $of2
done
