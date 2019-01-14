# -*- coding: utf-8 -*-
import numpy as np
from BootstrapFS import Define_DCM_Problem as Definition

"""
Created on Sun Sep 25 11:52:01 2016

FilterSmoother is a class that inherits from Define_Problem and uses it to sample from the dynamics defined in it. It needs as inputs
    -the number_of_particles 
    -timesteps_btw_obs
and initializes
    -the storage array for the filter particles: Filter
    -the storage array for the smoothing particles: Smoother
    -the integration step dt
Then, it samples particles using the sample() function and resamples using the weights estimated with the negLog_p() function. 

@author: HCRuiz
"""

class FilterSmoother(Definition):
    
    def __init__(self, number_of_particles, timesteps_btw_obs):
        self.timesteps_btw_obs = timesteps_btw_obs
        
        super(FilterSmoother, self).__init__(number_of_particles)
        
        self.dt = self.t_obs[1]/timesteps_btw_obs
        self.total_timepoints = int(self.T*(timesteps_btw_obs/self.t_obs[1]) + 1)
        self.Filter = np.zeros([self.number_of_particles,self.dim,self.total_timepoints])
        self.Smoother = np.zeros([self.number_of_particles,self.dim,self.total_timepoints])
        self.Bold_Signal = np.zeros([self.number_of_particles,self.total_timepoints])
        #print "t_obs[0] is ", self.t_obs[0], "giving condition", np.isclose(self.t_obs[0],self.t)
        
        
            
    def forward_pass(self):
        
        self.LogLikelihood = np.zeros(self.number_of_particles)
        #print "Mean LogLikelihood: ",np.mean(self.LogLikelihood)
        self.t = 0.
        self.sm_index = 0
        current_particles = self.X_initial
        if np.isclose(self.t_obs[0],0.):
            self.Cost = self.negLog_p(self.X_initial,self.t)
            #print "Cost:", self.Cost
        else:
            self.Cost = np.zeros(self.number_of_particles)
        
        for i in range(len(self.t_obs)-1):
            current_particles = self._resample(current_particles) # resample partices before propagating them to the next observation; it returns the current resample state of particles
            current_particles = self._sample(current_particles,i+1)
            
        current_particles = self._resample(current_particles)
        self.Filter[:,:,-1] = current_particles
        self.Smoother[:,:,-1] = current_particles
        
    def _resample(self, X):
        
        #compute weights
        w_pf = np.exp(-self.Cost)
        w_pf /= np.sum(w_pf)
#        print "sum of w_pf: ", np.sum(w_pf), "max w_pf:", np.max(w_pf), "min w_pf: ", np.min(w_pf)
        
        #Structural resampling
        u = np.arange(1.,self.number_of_particles+1)/self.number_of_particles
        bins = np.zeros(len(w_pf))
        bins[1:] = np.cumsum(w_pf,dtype=float)[:-1]
        index = np.digitize(u,bins,right=False)-1
        #print "index=",index
        #update current particles after resampling
        current_particles = X[index,:] 
        # Get cost of each resampled particle
        mll = np.mean(self.LogLikelihood)
        self.LogLikelihood = self.LogLikelihood - self.Cost[index] 
        #print "Mean LogLikelihood: ",np.mean(self.LogLikelihood)
        #assert mll>np.mean(self.LogLikelihood), "logL increased!"

        #reset cost
        self.Cost = np.zeros(self.number_of_particles)
        
        #Update smoothing particles
        #print "time index for resampling smoother: ",int(self.t/self.dt)
        #print "time index for observation: ",int(self.t_obs[np.isclose(self.t_obs,self.t)]/self.dt)
        #self.sm_index = int(self.t_obs[np.isclose(self.t_obs,self.t)]/self.dt)
        self.Smoother[:,:,:self.sm_index+1] = self.Smoother[index,:,:self.sm_index+1]
        
        return current_particles
    
    def _sample(self, current_particles, next_obs_index):
        '''Samples particles forward between observations; it initializes the particles at current_particles and propoages them until t_obs[i+1]. It stores the paticles in Filter array
        '''
        #saves current particles in the filter
        self.Filter[:,:,(next_obs_index-1)*self.timesteps_btw_obs] = current_particles
        self.Smoother[:,:,(next_obs_index-1)*self.timesteps_btw_obs] = current_particles
        #propagates particles between observations
        for i in range(1,self.timesteps_btw_obs+1):
            current_particles = current_particles + self._one_step(current_particles)*self.dt
            self.Filter[:,:,(next_obs_index-1)*self.timesteps_btw_obs+i] = current_particles
            self.Smoother[:,:,(next_obs_index-1)*self.timesteps_btw_obs+i] = current_particles
            self.t += self.dt
            self.sm_index += 1
        
        self.Cost = self.negLog_p(current_particles,self.t)
        
        return current_particles
        
    def _one_step(self, X):
        
        noise = np.random.randn(X.shape[0],X.shape[1])/np.sqrt(self.dt) # noise needs to have variance 1/dt
        update_step = self.Drift(X) + self.Diffusion(X)*noise
        
        return update_step
    
    def compute_statistics(self):
        # Get Log-Likelhood of particles resampled
        #self.Likelihood = np.exp(self.LogLikelihood)
        self.mean_logL = np.mean(self.LogLikelihood)
        # Compute Bold signal from obtaind smoothing particles
        for t in range(self.total_timepoints):
            self.Bold_Signal[:,t] = self.obs_signal(self.Smoother[:,:,t],t)[:,0]
            
        # Get particle statistics
        self.mean_Filter = np.mean(self.Filter,axis=0)
        self.mean_Smoother = np.mean(self.Smoother,axis=0)
        self.mean_Bold_Signal = np.mean(self.Bold_Signal,axis=0)
        self.var_Filter = np.mean(self.Filter**2,axis=0)-np.mean(self.Filter,axis=0)**2
        self.var_Smoother = np.mean(self.Smoother**2,axis=0)-np.mean(self.Smoother,axis=0)**2  
        self.var_Bold_Signal = np.mean(self.Bold_Signal**2,axis=0) - np.mean(self.Bold_Signal,axis=0)**2
        
        