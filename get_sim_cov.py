# Do we need to subtract mean-field for covariance? Not sure
# We do here anyway

import matplotlib
matplotlib.use('Agg')
from os.path import join as opj, dirname
import os
from ksz4.cross import four_split_K, split_phi_to_cl
from pixell import enmap, curvedsky
from scipy.signal import savgol_filter
from cmbsky import safe_mkdir, get_disable_mpi, ClBinner
from falafel import utils, qe
import healpy as hp
import yaml
import argparse
from orphics import maps
import numpy as np
import matplotlib.pyplot as plt

import pickle


import sys
sys.path.append("../plots")
from plot_tools import get_CLKK_stuff
sys.path.append("../bin")
from run_auto import get_weight_map

def plot_e2e_sims(axs, output_path_template, meanfield_dir, rdn0_file, binner, est="qe", nsim=64):
    ax,ax_hist=axs
    frac_diffs=[]
    bias_over_sigs=[]
    amps = []
    cl_kks = []

    nsim_read=0
    for irot in range(nsim):
        print("isim:",irot)
        d=output_path_template%irot
        try:
            with open(opj(d, "auto_outputs.pkl"), "rb") as f:
                pickle.load(f)
        except Exception as e:
            print(e)
            print("skipping irot=%d"%irot)
            continue
        nsim_read+=1

        cl_kk_stuff = get_CLKK_stuff(
                d, meanfield_dir, rdn0_file, w1, w4, binner, use_mcn0=True,
                est=est
        )
        cl_kks.append(cl_kk_stuff["CL_KK"])
        bias_over_sigs.append( cl_kk_stuff["bias_over_sig"] )
        frac_diffs.append( cl_kk_stuff["frac_diff"] )

        #ax.plot(binner.bin_mids, CL_KK / binner(auto_outputs["Cl_KK_ksz_theory"])-1)
        ax.plot(binner.bin_mids, binner(cl_kk_stuff["bias_over_sig"]))

        amps.append(cl_kk_stuff["A_ksz"])

        print("A_ksz = %.6f +/- %.6f"%(cl_kk_stuff["A_ksz"], cl_kk_stuff["A_ksz_err"]))
        print("binned A_ksz = %.6f +/- %.6f"%(cl_kk_stuff["A_ksz_binned"], cl_kk_stuff["A_ksz_binned_err"]))


    #ax.plot(binner.bin_mids, binner.bin_mids**2*binner(auto_outputs["Cl_KK_ksz_theory"]), color="k")
    frac_diffs = np.array(frac_diffs)
    bias_over_sigs = np.array(bias_over_sigs)

    mean_frac_diff = frac_diffs.mean(axis=0)
    mean_frac_diff_err = np.std(frac_diffs, axis=0)/np.sqrt(nsim_read-1)

    #ax.fill_between(binner.bin_mids, mean_frac_diff-mean_frac_diff_err, mean_frac_diff+mean_frac_diff_err, 
    #               color="k", alpha=0.25)

    ax_hist.hist(np.array(amps)-1)
    ax_hist.set_xlabel("$A_{ksz} - 1$")
    A_mean, A_err = np.mean(amps), np.std(amps)/np.sqrt(nsim)
    ax_hist.set_title(r"$\bar{A}_{ksz} = %.3f \pm %.3f, \sigma(A)=%.3f$"%(A_mean, A_err, np.std(amps)))

    #ax.set_yscale('symlog', linthreshy=10.)
    ax.set_xscale('log')
    ax.set_xlabel("$L$")
    ax.set_ylabel("bias/sigma in $C_L^{KK}$")
    
    return amps, A_mean, A_err, cl_kks

OUTPUTS_DIR="/pscratch/sd/m/maccrann/ksz_outputs/sim_e2e_test/"

def main(args):
    
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        comm = None
        rank,size = 0,1

    args.mask=args.mask_file
    total_mask, extra_mask = get_weight_map(args, verbose=(rank==0))
    w1 = maps.wfactor(1,total_mask)
    w4 = maps.wfactor(4,total_mask)

    fig,axs=plt.subplots(ncols=2, figsize=(8,4))
    nsim=128
    
    cl_kks = []
    nsim_read=0

    K_AB_sum = 0
    K_CD_sum = 0
    
    if args.est=="qe":
        K_AB_name = "K_ab.npy"
        K_CD_name = "K_cd.npy"
    else:
        K_AB_name = "K_ab_%s.npy"%args.est
        K_CD_name = "K_cd_%s.npy"%args.est
    
    print("getting meanfield")
    for i in range(nsim):
        if size>1:
            if i%size != rank:
                continue
        print(i)
        d = "/pscratch/sd/m/maccrann/ksz_outputs/sim_e2e_test/"+args.tag+"/output_rot%d"%i
        try:
            with open(opj(d, "auto_outputs.pkl"), "rb") as f:
                pickle.load(f)
        except Exception as e:
            print(e)
            print("skipping irot=%d"%i)
            continue
        
        K_AB_sum += np.load(opj(d, K_AB_name))
        K_CD_sum += np.load(opj(d, K_CD_name))
            
        nsim_read+=1
        
    if rank==0:
        n_collected=1
        while n_collected<size:
            K_AB_sum_recv, K_CD_sum_recv, nsim_read_recv = comm.recv(source=MPI.ANY_SOURCE)
            K_AB_sum += K_AB_sum_recv
            K_CD_sum += K_CD_sum_recv
            nsim_read += nsim_read_recv
            n_collected+=1

        K_AB_mf = K_AB_sum/nsim_read
        K_CD_mf = K_CD_sum/nsim_read
        print("read %d sims"%nsim_read)
    else:
        K_AB_mf, K_CD_mf = 0.,0.
        comm.send((K_AB_sum,K_CD_sum, nsim_read),
                  dest=0)
    comm.Barrier()
    K_AB_mf, K_CD_mf = comm.bcast((K_AB_mf, K_CD_mf), root=0)

    
    nsim_read = comm.bcast(nsim_read, root=0)
    
    print("K_AB_mf.shape:",K_AB_mf.shape)
    print("nsim_read:", nsim_read)
    
    #Now loop through getting auto
    print("now getting sim CL_KKs")
    nsim_read_second_time = 0 #just paranoid checking
    for i in range(nsim):
        if size>1:
            if i%size != rank:
                continue
        print(i)
        d = "/pscratch/sd/m/maccrann/ksz_outputs/sim_e2e_test/"+args.tag+"/output_rot%d"%i
        try:
            with open(opj(d, "auto_outputs.pkl"), "rb") as f:
                pickle.load(f)
        except Exception as e:
            print(e)
            print("skipping irot=%d"%i)
            continue        
        K_AB = np.load(opj(d, K_AB_name))
        K_CD = np.load(opj(d, K_CD_name))
        #when we subtract the mean-field, we're subtracting off
        #the signal/nsim_read. So need to divide by (1-1./nsim_read)
        K_AB_mfcorrected = (K_AB - K_AB_mf)/(1.-1./nsim_read)
        K_CD_mfcorrected = (K_CD - K_CD_mf)/(1.-1./nsim_read)
        cl_kks.append(
            split_phi_to_cl(K_AB_mfcorrected,
                             K_CD_mfcorrected)/w4
        )
        nsim_read_second_time+=1
        
    if rank==0:
        n_collected=1
        while n_collected<size:
            cl_kks_recv, nsim_read_second_time_recv = comm.recv(source=MPI.ANY_SOURCE)
            cl_kks += cl_kks_recv
            nsim_read_second_time += nsim_read_second_time_recv
            n_collected+=1
        assert len(cl_kks)==nsim_read_second_time==nsim_read
        cl_kks = np.array(cl_kks)
        print(cl_kks.shape)
        
        filename=opj(OUTPUTS_DIR, args.tag, "%s_CLKKs_nsim%d.npy"%(args.tag, args.nsim))
        print("saving cl_kks to %s"%filename)
        np.save(filename, cl_kks)
        
    else:
        comm.send((cl_kks,nsim_read_second_time), dest=0)
        return 0

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="run tests")
    parser.add_argument("tag", type=str)
    parser.add_argument("--meanfield_dir", type=str, default="mean_field_nsim64")
    parser.add_argument("--rdn0_file", type=str, default="rdn0_outputs_nsim32.pkl")
    parser.add_argument("--nsim", type=int, default=128)
    parser.add_argument("--est", type=str, default="qe")
    parser.add_argument("--mask_file", type=str, default="/global/cfs/projectdirs/act/data/synced_maps/DR6_lensing/masks/act_mask_20220316_GAL060_rms_60.00sk.fits")
    parser.add_argument("--apply_extra_mask", type=str, default=None)
    parser.add_argument("--extra_mask_power", type=float, default=None)
    parser.add_argument("--smooth_extra_mask", type=float, default=None)
    parser.add_argument("--mpi", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
    #get_N1(args.nsim, use_mpi=args.mpi)
