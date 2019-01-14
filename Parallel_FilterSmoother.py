# -*- coding: utf-8 -*-
import sys
from time import time
import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt 
"""
Created on Sun Sep 25 17:06:46 2016
The Parallel_FilterSmoother generates Filter and Smoothing estimates in parallel over multiple workers and reduces them to a single estimate of the mean and variance of the smoothing distribution by averaging over the worker estimates.
@author: HCRuiz
"""

from BootstrapFS import FilterSmoother

comm = MPI.COMM_WORLD

def savedata(var_name,data):
    if comm.Get_rank() == 0 and len(sys.argv)>2:
        file_id = sys.argv[2]+"_"+var_name
        print file_id+" saved!"
        np.save(file_id,data)
        
iterations = 20
number_of_particles = 8000
timesteps_btw_obs = 40
#Initialize FS
FS = FilterSmoother(number_of_particles,timesteps_btw_obs)

if comm.Get_rank()==0:
    print 'Number of observations: ', FS.Y.shape[0]
    print 'Dimension of observations: ', FS.dim_obs
    print 'Number of workers: ', comm.Get_size()
    
#Initialize aux. variables
mean_Smoother_wrk = np.zeros([FS.dim,FS.total_timepoints])
var_Smoother_wrk = np.zeros([FS.dim,FS.total_timepoints])
mean_Filter_wrk = np.zeros([FS.dim,FS.total_timepoints])
var_Filter_wrk = np.zeros([FS.dim,FS.total_timepoints])
mean_Bold_Signal_wkr = np.zeros(FS.total_timepoints)
var_Bold_Signal_wkr = np.zeros(FS.total_timepoints)
particle_Likelihood = np.zeros([iterations,number_of_particles])
mean_logL_wkr = np.zeros(iterations)

if comm.Get_rank()==0: start_time = time()
for i in range(iterations):
    if comm.Get_rank()==0: print "Iteration: ", i
    FS.forward_pass()
    FS.compute_statistics()
    
    mean_logL_wkr[i] = FS.mean_logL
    
    mean_Smoother_wrk += FS.mean_Smoother/iterations
    var_Smoother_wrk += FS.var_Smoother/iterations
    
    mean_Bold_Signal_wkr += FS.mean_Bold_Signal/iterations
    var_Bold_Signal_wkr += FS.var_Bold_Signal/iterations
    
    mean_Filter_wrk += FS.mean_Filter/iterations
    var_Filter_wrk += FS.var_Filter/iterations

    
allMeans_Bold_Signal = comm.gather(mean_Bold_Signal_wkr,root=0)
allVars_Bold_Signal = comm.gather(var_Bold_Signal_wkr,root=0)

allMeans_Smoother = comm.gather(mean_Smoother_wrk,root=0)
allVars_Smoother = comm.gather(var_Smoother_wrk,root=0)

allMeans_Filter = comm.gather(mean_Filter_wrk,root=0)
allVars_Filter = comm.gather(var_Filter_wrk,root=0)

allMeans_logL = comm.gather(mean_logL_wkr,root=0)
#print "Hi from worker ", comm.Get_rank()

if comm.Get_rank() == 0:
    particle_logLikelihood = FS.LogLikelihood
    allMeans_logL = np.array(allMeans_logL)
    
    allMeans_Bold_Signal = np.array(allMeans_Bold_Signal)
    allVars_Bold_Signal = np.array(allVars_Bold_Signal)
    
    allMeans_Smoother = np.array(allMeans_Smoother)
    allVars_Smoother = np.array(allVars_Smoother)
#    print "shape of allMeans: ", allMeans.shape
#    print "shape of allVars: ", allVars.shape
#    print allMeans[1,0,:]
    total_mean_Bold = np.mean(allMeans_Bold_Signal,axis=0)
    total_var_Bold = np.mean(allVars_Bold_Signal,axis=0)
    
    total_mean_Smoother = np.mean(allMeans_Smoother,axis=0)
    total_var_Smoother = np.mean(allVars_Smoother,axis=0)
    
    
    elapsed = (time() - start_time)
    
    allMeans_Filter = np.array(allMeans_Filter)
    allVars_Filter = np.array(allVars_Filter)
    total_mean_Filter = np.mean(allMeans_Filter,axis=0)
    total_var_Filter = np.mean(allVars_Filter,axis=0)
    
    print 'Elapsed time for ',iterations,' iterations: ', elapsed, 'sec'
    
    savedata('particle_Likelihood',particle_Likelihood)
    savedata('allMeans_logL',allMeans_logL)
    
    savedata('allMeans_Bold_Signal',allMeans_Bold_Signal)
    savedata('allVars_Bold_Signal',allVars_Bold_Signal)
    savedata('total_mean_Bold_Signal',total_mean_Bold)
    savedata('total_var_Bold_Signal',total_var_Bold)
    
    savedata('allMeans_Smoother',allMeans_Smoother)
    savedata('allVars_Smoother',allVars_Smoother)
    savedata('total_mean_Smoother',total_mean_Smoother)
    savedata('total_var_Smoother',total_var_Smoother)
    
    savedata('allMeans_Filter',allMeans_Filter)
    savedata('allVars_Filter',allVars_Filter)
    savedata('total_mean_Filter',total_mean_Filter)
    savedata('total_var_Filter',total_var_Filter)
    fMRI_series = FS.Y
    savedata('fMRI_series',fMRI_series)
    taxis = np.arange(total_mean_Smoother.shape[1])*FS.dt
    savedata('taxis',taxis)
    savedata('t_obs',FS.t_obs)
