# Code for the DR6 kSZ trispectrum measurement pipeline

This repository contains the pipeline code for generating the kSZ trispectrum measurements presented in [MacCrann et al. 2024](https://arxiv.org/abs/2405.01188).

The scripts in this repository use a library code [ksz4](https://github.com/simonsobs/ksz4/tree/main), so start by installing that.

The rest of the complexity in the pipeline mainly comes down to pre-processing the DR6 data (in our case we use a harmonic-space ILC), and running bias corrections using simulations (which also need to be pre-processed/generated). I haven't ported the code for that into this repository yet, but see the README in the preprocess folder for more info. 


# Running the pipeline

We use a couple of scripts:
1. `scripts/run_auto.sh` runs the measurement and bias corrections. 
2. `sim_e2e_test/run_e2e_sim.sh` runs the "end-to-end" simulations used for the transfer function correction and covariance matrix. 
`sim_e2e_test/run_e2e_sim.sh`
