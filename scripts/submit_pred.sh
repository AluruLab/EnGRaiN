for x in `cat tc.txt`
do
    for y in 3 4 5a 5b 6
    do
      dir=ensemble_preds${y}
      #echo qsub run_template.pbs -N m${y}_${x}_run -v "tissue=$x,method=$y,odir=$dir"
      #qsub run_template.pbs -N m${y}_${x}_run -v "tissue=$x,method=$y,odir=$dir"
      echo qsub threshold_preds.pbs -N th${y}_${x}_run -v "tissue=$x,method=$y,odir=$dir"
      qsub threshold_preds.pbs -N th${y}_${x}_run -v "tissue=$x,method=$y,odir=$dir"
    done
done
