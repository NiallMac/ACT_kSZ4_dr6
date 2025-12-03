#!/bin/bash

#Run estimator on many sims and compute covariance matrix
#We do not re-run mean-field and rdn0 for every simulation.
#Use the mean-field and RDN0 from the data measurement...

#source /global/common/software/act/maccrann/lenspipe/bin/activate
source /global/cfs/projectdirs/act/data/maccrann/lenspipe_py3.13/bin/activate
export DISABLE_MPI=false

for config in  dr6v4_v4_lmax4000_mask60sk_noisysims_re-run_newpipe # auto_config_lmax5000_lh auto_config_lmax6000 
do
    
    config_file=configs/${config}.yml
    for est_maps in hilc_hilc_hilc_hilc
    do

#output directory for data measurement (will use mean-field and rdn0 from here)
data_measurement_output_dir="/pscratch/sd/m/maccrann/ksz_outputs/output_hilc_hilc_hilc_hilc_dr6v4_v4_lmax4000_mask60sk_noisysims_re-run"
get_sim_cov_extra_args="--meanfield_dir ${data_measurement_output_dir}/mean_field_nsim64 --rdn0_file ${data_measurement_output_dir}/rdn0_outputs_nsim64.pkl"

#where the simulations are
sim_dir="/pscratch/sd/m/maccrann/cmb/act_dr6/ilc_cldata_smooth-301-2_v4_lmax6000_60skmask_31102025/data_signal_sims"
#template for the "data" which in this case is simulation
data_template_path="${sim_dir}/sim_planck%s_act00_%s/\${freq}_split\${split}_wksz_rot%s.fits"

nrot=128

outdir_base="/pscratch/sd/m/maccrann/ksz_outputs/sim_cov/output_${config}_${est_maps}"
outdir="${outdir_base}/output"

echo "running auto realizations"
echo `date` >> $logfile 2>&1
#now run other realizations
cmd="srun -u -l -n 16 python ../bin/run_auto_rots.py ${outdir} --data_template_path $data_template_path --mask $mask -c ${config}.yml --est_maps $est_maps --nrot $nrot"
echo $cmd >> $logfile 2>&1
$cmd >> $logfile 2>&1
echo `date` >> $logfile 2>&1

#and get CL_KK realizations
cmd="srun -u -l -n 8 python get_sim_cov.py output_${config}_${est_maps} --mpi --nsim $nrot $get_sim_cov_extra_args"
echo $cmd >> $logfile 2>&1
$cmd >> $logfile 2>&1

done
done
