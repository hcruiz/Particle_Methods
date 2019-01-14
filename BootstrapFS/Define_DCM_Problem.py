# -*- coding: utf-8 -*-
import sys
import os
sys.path.append("../../")
import numpy as np
import scipy.io as sio
"""
Created on Sun Sep 25 09:10:16 2016

This class Define_Problem defines problem for the Bootstrap Particle Filter-Smoother implemented in a class called Parallel_FilterSmoother.

This class contains:
    -A function loading the data
    -Definition of the model parameters: dimension, time horizon, dyn. and observation noise variance, etc
    -Defines needed functions for dynamics and cost: drift, diffusion, negLog_likelihood
    -Defines initialization of particles

These are then inherited to Parallel_FilterSmoother where a rollout, a compute_weights and a resampling functions are defined

@author: HCRuiz
"""

class Define_DCM_Problem(object):
    #####################################
    ## Algorithmic specific parameters ##
    #####################################
    
    mvn_prior = True
    
    ####################################
    ######## Helper Functions ##########
    ####################################
    def load_data(self):
        ''' Loads data from files and defines the variables Y, t_obs and eventually var_obs
        self.Y = np.array([[0.],[5.],[-0.5]])
        self.t_obs = np.array([0.,1.,2.])
        self.dim_obs = self.Y.shape[1]
        
#        print 'Number of observations: ', self.Y.shape[0]
#        print 'Dimension of observations: ', self.dim_obs
        '''
        trial = 1    
        Subject = '1'
        ROI = 'V'
        BOLD_TS = 'BOLD_'+ROI
        ts = int(sys.argv[1]) #int(ts_identifier)    

        #Load time-series
        dir_data = 'Data_SmoothingProblem/SUBJECT'+Subject+'/BOLD_TS_'+ROI+'.mat'
        #print os.getcwd()
        
        time_Series = sio.loadmat(dir_data) #the time series is a directory like object with key 'tobs' for the time points and key BOLD_TS for the observed signal
        self.t_obs = time_Series['tobs'][0,:] # The time points have shape [no. of observations,]
        print 'No. of observations ', self.t_obs.shape[0]
        print 'Time series: ', BOLD_TS, 'in ',dir_data, ' loaded'
        print 'Initialize fMRI smoother for time series #',ts
        
        ### fMRI time series ###
        Y = time_Series[BOLD_TS][ts,:,trial-1] # time_Series[BOLD_TS] is an TSx(# of observ)x(trials)-array containing all fMRI of one subject. Each trial has TS events giving a fMRI time series of length (# of observ).
        try: self.dim_obs = Y.shape[1] # dimensions of the observed signal
        except : self.dim_obs = 1
        # The data has to have shape [no. of observations,dim_obs] print 'shape of data: ', self.Y.shape
        self.Y = Y.reshape([len(Y),self.dim_obs])   
        
        ### Observation noise estimated from fMRI time series ###
        self.var_obs = (time_Series['std_'+BOLD_TS][ts,trial-1])**2
        
    def _set_initial_conditions(self):
        
        if self.mvn_prior:#the initial conditions can be set for each rollout on this core independently.
            #print 'mvn_prior'
            self.X_initial = np.random.multivariate_normal(self.mean_prior,self.cov_prior,self.number_of_particles)            
        else:
            #print 'Fixed prior as default'
            self.X_initial = np.zeros([self.number_of_particles,self.dim]) #the initial conditions. They can be set for each rollout on this core independently.
            self.X_initial[:,-3:] = 1
    
    #####################################
    ##### Model specific parameters #####
    #####################################
    #obs_dim = 0
    dim = 5 # X = [z,s,f,v,q]
    E_0=0.4
    A=1.0#0.0
    sigma_dyn = np.zeros([dim,1])
    sigma_dyn[0] = 0.15**2
    
    ### PRIOR ###
    mean_prior = np.array([0.,0.,1.,1.,1.])
    sigma_prior = np.array([0.016,0.01,0.019,0.006,0.006]) 
    cov_prior = np.diag(sigma_prior**2)

    #####################################
    ##### Model specific functions  #####
    #####################################    

    def Drift(self,state):
        # X = [z,s,f,v,q]
        Drift = np.zeros_like(state)
        # Neuronal Activity z
        B=0.
        C=0.
        t_on = np.array([0.])# np.arange(5.0,T,10.) #
        t_off = np.array([0.])# np.arange(5.15,T,10.) #

        #def I(t):
            #global t_inp
        #    inp = 0.0
        #    if t>=t_on[self.t_inp] and t<t_off[self.t_inp]:
        #        inp = 1.0
        #    if np.isclose(t,t_off[self.t_inp]) and t < t_off[-1]:
        #        self.t_inp += 1
        #    return inp

        z = state[:,0]
        inp = 0.#I(t)
        z_dot = (-1.0 + B*inp)*z + C*inp

        # Hemodynamic system: parameters for s 
        epsilon=0.8
        tau_s=1.54
        tau_f=2.44

        s = state[:,1]
        f = np.absolute(state[:,2])
        
        s_dot = epsilon*z - s/tau_s - (f - 1)/tau_f
        f_dot = s

        # Balloon model: v and q
        tau_0=1.02
        alpha=0.32
        alpha=1.0/alpha

        v = np.absolute(state[:,3])
        q = state[:,4]

        v_dot = (f - v**(alpha))/tau_0
        q_dot = ( f*(1-(1-self.E_0)**(1/f))/self.E_0 - (v**(alpha-1))*q )/tau_0

        Drift = np.zeros_like(state)
        Drift[:,0] = self.A*z_dot
        Drift[:,1] = s_dot
        Drift[:,2] = f_dot
        Drift[:,3] = v_dot
        Drift[:,4] = q_dot

        return Drift

    
    def Diffusion(self,x):
    
        return self.A*self.sigma_dyn.T#*(x+1)
    
    #Defines the Cost    
    def negLog_p(self,state, t):
        '''    
#        print "shape of x:", x.shape
        if np.any(np.isclose(t,self.t_obs)):
            #print "observation:",self.Y[np.isclose(t,self.t_obs),self.obs_dim]
            return 0.5*(x[:,self.obs_dim]-self.Y[np.isclose(t,self.t_obs),self.obs_dim])**2/self.var_obs
        else:
            print 'Cost is zero!'
            return np.zeros(x.shape[0])
        '''
        
        neg_logLikelihood = np.zeros([self.number_of_particles,self.dim_obs])
        check = np.isclose(t,self.t_obs)
        if np.any(check):
            #print 'observ time:', t
            neg_logLikelihood = (self.obs_signal(state,t)-self.Y[check,:])**2/(2*self.var_obs)
            #print 'shape(obs_signal): ', np.shape(obs_signal(state,t))
            #print 'shape(Y): ', np.shape(self.Y[check])
            #print 'shape(neg_log): ', np.shape(neg_logLikelihood)
        return neg_logLikelihood
    
    def obs_signal(self,state, t):
        state_obs = state[:,-2:] # this is the states on which the signal depends;  doesn't have to be the same dimensions as the obs_signal: for DCM the dynamic variables are v and q ( X = [z,s,f,v,q])
        v = state_obs[:,0]
        q = state_obs[:,1]

        Vo = 0.018
        k1=7.0*self.E_0
        k2=2.0
        k3=2.0*self.E_0-0.2

        BOLD = Vo*( k1*(1.-q) + k2*(1.-q/v) + k3*(1.-v) )

        return BOLD.reshape(np.shape(state_obs)[0],1)

    def __init__(self,number_of_particles):
        self.load_data()
        self.T = self.t_obs[-1] #defines horizon time
        self.number_of_particles = number_of_particles
        self._set_initial_conditions()
        
