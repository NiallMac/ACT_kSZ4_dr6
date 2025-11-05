#!/bin/bash

#setup your environment 
source /global/cfs/projectdirs/act/data/maccrann/lenspipe_py3.13/bin/activate
export DISABLE_MPI=false
export MPI4PY_RC_RECV_MPROBE=0

#config file tag
tag=dr6v4_v4_lmax4000_mask60sk_noisysims_re-run
    
config=auto_configs/${tag}.yml

#output directory 
outdir="/pscratch/sd/m/maccrann/ksz_outputs/"

for est_maps in hilc_hilc_hilc_hilc  #frequency (one for each trispectrum leg), you could loop over multiple frequency combination options here
do
    echo $est_maps
    echo `date`

    #run the auto measurement (multiple processes help for the cross-estimator)
    cmd="srun -u -l -n 8 python bin/run_auto.py ${outdir}/output_${est_maps}_${tag} -c $config --use_mpi True --est_maps $est_maps"
    echo $cmd
    $cmd

    #rdn0 and meanfield (same script but now use more processees since we're iterating over sims)
    cmd="srun -u -l -n 32 python bin/run_auto.py ${outdir}/output_${est_maps}_${tag} -c $config --use_mpi True --do_rdn0 True --do_meanfield True  --skip_auto True --est_maps $est_maps"
    echo $cmd
    $cmd
    echo `date`
done

