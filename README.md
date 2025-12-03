# Code for the DR6 kSZ trispectrum measurement pipeline

This repository contains the pipeline code for generating the kSZ trispectrum measurements presented in [MacCrann et al. 2024](https://arxiv.org/abs/2405.01188).

The scripts in this repository use a library code [ksz4](https://github.com/simonsobs/ksz4/tree/main), so start by installing that.

The rest of the complexity in the pipeline mainly comes down to pre-processing the DR6 data (in our case we use a harmonic-space ILC), and running bias corrections using simulations (which also need to be pre-processed/generated). I haven't ported the code for that into this repository yet, but see the README in the preprocess folder for more info. 


## Running the pipeline

There are a few steps:
1. Run the measurement and bias corrections. 
2. Run the "end-to-end" simulations used for the transfer function correction and covariance matrix. 
`sim_e2e_test/run_e2e_sim.sh`
3. Plotting/reporting statistics etc. 

### 1. Run the measurement and bias corrections.

For this I used `run_auto.sh` for this which I ran on 4 interactive nodes at NERSC perlmutter:
```
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

for est_maps in hilc_hilc_hilc_hilc  #frequency option, you could loop over multiple frequency options here
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
```
There are two calls to the python script `bin/run_auto.py`, which 
- Sets up options for the measurement (based on the config file, here `auto_configs/dr6v4_v4_lmax4000_mask60sk_noisysims_re-run.yml`)
- Runs the data measurement (i.e. applies the quadratic estimators we've setup to the data alms)
- Is called again to run the RDN0 and meanfield. On reflection this could actually all be done in one call of `bin/run_auto.py`. You'd just have more processes than needed for the auto measurement step,
  but that shouldn't cause issues I don't thin...
This will save to `${outdir}/output_${est_maps}_${tag}`
- `K_ab.npy` and `K_cd.npy` are the K maps. Actually, they are arrays of K maps necessary for the cross-correlation-only $C_L^KK$ estimator. `K_ab.npy` is formed from the first two maps specified, and `K_cd.npy` from the second two. In this case, all the frequencies are  the same ("hilc_hilc_hilc_hilc"), so these two maps are identical. But in other cases we use e.g. the `hilc_hilc-tszandcibd_hilc_hilc` frequency combination, which labels the case where the second leg has tSZ and CIB deprojected, in this case `K_ab.npy` and `K_cd.npy` would be different.

  
- `rdn0_outputs_nsim64.pkl` a dictionary, saved as a pkl, containing the outputs of the rdn0 calculation. You can read this in and e.g. access the RDN0 as the `rdn0` key.
- `mean_field_nsim64` is a directory containing all the reconstructions used to calculate the mean-field (which could be deleted now), and their mean i.e. the mean-field, saved as `Ks_ab_mean.pkl` and `Ks_cd_mean.pkl`

### 2. Run the covariance.

To get the covariance, we run the $C_L^{KK}$ measurement on noisy simulations. We include in these simulations a randomly rotated realisation of the Alvarez 2016 kSZ signal (although it is probably negligible). Due to computational limitations, we don't do RDN0 (hence I think as far as the covariance is concerned, we don't need to do any N0 correction at all, since it would be the same for all realisations). There is a bash script ./run_cov.sh that runs this:

```
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
```

