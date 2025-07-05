import os
# Redundant if you already exported above, but safe to repeat here:
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=32'
os.environ['OMP_NUM_THREADS'] = '1'

from numpyro_model import *
from model_new import *

import numpy as np
import scipy as sp
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import jax.scipy as jsp
import pickle
jax.config.update("jax_enable_x64", True)


print("local_device_count():", jax.local_device_count())
numpyro.set_host_device_count(16)


Vc0 = 240

df_mockdata = pd.read_csv('/data/hz420-2/radial_migration_kernel/mock_sample/L_conditioned/mock_data_errorfree_L@10.csv')
F, logage, L = df_mockdata['MH'], df_mockdata['log_age'], df_mockdata['Lz']
sigma_F, sigma_logage, sigma_L = df_mockdata['sigma_MH'], df_mockdata['sigma_logage'], df_mockdata['sigma_Lz']

e_feh_median = np.amax([np.median(sigma_F), 0.01])
e_log10age_median = np.amax([np.median(sigma_logage), 0.02])
e_Lz_median = np.amax([np.median(sigma_L), 20])
print('Median errors: e_feh_median = ', e_feh_median,
      'e_log10age_median = ', e_log10age_median,
      'e_Lz_median = ', e_Lz_median)
feh_range = [-1.5,0.5]
logage_range = np.log10([0.3,12])
L_range = [9*Vc0,11*Vc0]
nfe = int((feh_range[1] - feh_range[0]) / e_feh_median)
nlogage = int((logage_range[1] - logage_range[0]) / e_log10age_median)
nL = int((L_range[1] - L_range[0]) / e_Lz_median)
print('Fe/H bins: ', nfe, 'bin size: ', (feh_range[1] - feh_range[0]) / nfe)
print('log10(age) bins: ', nlogage, 'bin size: ', (logage_range[1] - logage_range[0]) / nlogage)
print('Lz bins: ', nL, 'bin size: ', (L_range[1] - L_range[0]) / nL)

data_array = np.array([F, logage, L]).T
H, edges = np.histogramdd(data_array, bins=(nfe, nlogage, nL), range=[feh_range, logage_range, L_range])
print('H shape: ', H.shape, 'Total stars: ', H.sum(), 'Number of bins with H>1', (H>=1).sum())
# print(edges)
centers = [
    0.5*(edges[d][:-1] + edges[d][1:]) 
    for d in range(3)
]

ix, iy, iz = np.nonzero(H)  # each is a 1D array of the same length M

fe_centers     = centers[0][ix]
logage_centers = centers[1][iy]
L_centers      = centers[2][iz]
Nstars = H[ix, iy, iz]

print('Final MH range:', fe_centers.min(), fe_centers.max())
print('Final age range:', 10**logage_centers.min(), 10**logage_centers.max())
print('Final Lz range:', L_centers.min(), L_centers.max())  
# print(len(fe_centers), len(logage_centers), len(L_centers)
#       , len(H[ix, iy, iz]), H[ix, iy, iz].sum())
# plt.figure()
# cb = plt.scatter(10**logage_centers, fe_centers, c=H[ix, iy, iz], cmap='viridis', alpha=0.5)
# plt.colorbar(cb, label='Number of stars')
# plt.savefig('/data/hz420-2/radial_migration_kernel/mock_sample/L_conditioned/mock_data_binned.png', dpi=300)
# plt.show()


data_grid = {
    'MH': jnp.array(fe_centers),
    'log_age': jnp.array(logage_centers),
    'Lz': jnp.array(L_centers),
    'sigma_MH': jnp.array([e_feh_median] * len(fe_centers)),
    'sigma_logage': jnp.array([e_log10age_median] * len(logage_centers)) * 0,
    'sigma_Lz': jnp.array([e_Lz_median] * len(L_centers)),
}

R_scale_for_sampling = 4
F_centre_for_sampling, F_scale_for_sampling = -0.5,  -0.5

data_generated = generate_sample_for_MC_integration(data_grid, R_scale_for_sampling = R_scale_for_sampling, 
                                                    F_centre_for_sampling = F_centre_for_sampling, 
                                                    F_scale_for_sampling = F_scale_for_sampling, 
                                                    N_sample = int(1e3)) 

data_generated['weights'] = jnp.array(Nstars)


Nknots = 10
aux_params = {'R_scale_for_sampling':R_scale_for_sampling, 
              'F_scale_for_sampling':F_scale_for_sampling, 
              'F_centre_for_sampling':F_centre_for_sampling}  # Auxiliary parameters for the model
aux_params['aux_knots'] = generate_aux_knots(Nknots=Nknots, age_max=12.)#jnp.linspace(0.,15.,Nknots)
print('aux knots:', aux_params['aux_knots'])
# aux_params = {}

prior_range = prior_scale_uniform # Prior is set in model_new.py
parameters = {
    'ln_Rdisk': numpyro.distributions.Uniform(low=prior_range['ln_Rdisk'][0]*jnp.ones(Nknots), high=prior_range['ln_Rdisk'][1]*jnp.ones(Nknots),).expand([Nknots]),
    'ln_sigmaLz': numpyro.distributions.Uniform(low=prior_range['ln_sigmaLz'][0]*jnp.ones(Nknots), high=prior_range['ln_sigmaLz'][1]*jnp.ones(Nknots),).expand([Nknots]),}

def logL_zero(params, data):
    # Return the log likelihood
    return 0


init_from_minimiser = True

n_warmup = 400
n_samples = 2000
num_chains = 1
max_tree_depth = 7
target_accept_prob = 0.9
step_size = 1e-2
adapt_step_size = True
extra_fields = ('num_steps', 'adapt_state.step_size')
jit_model_args = True
# init_strategy=numpyro.infer.init_to_sample()

if init_from_minimiser:

    file_name = f'/data/hz420-2/radial_migration_kernel/mock_sample/L_conditioned/minimization_result_{Nknots}knots_L@10.npy'
    minimiser_results = np.load(file_name)
    init_guess = {
        'ln_Rdisk': jnp.array(minimiser_results[:Nknots]),
        'ln_sigmaLz': jnp.array(minimiser_results[Nknots:2*Nknots]),
    }
    print('Initial guess from minimiser:', init_guess)
    init_strategy=numpyro.infer.initialization.init_to_value(values=init_guess)
else:
    init_strategy=numpyro.infer.init_to_sample()



print('model initialising...')

model = numpyro_model(logL_numpyro, parameters, data_generated, aux_parameters=aux_params, 
                      expand_args=True, log_prior_fn=log_prior)#logL_numpyro, logL_zero

print('model initialised, and start running MCMC...')
model.run_mcmc(num_warmup=n_warmup, num_samples=n_samples, num_chains=num_chains, 
                   init_strategy=init_strategy, max_tree_depth=max_tree_depth, step_size=step_size,
                   target_accept_prob=target_accept_prob, adapt_step_size=adapt_step_size,
                   chain_method="sequential", extra_fields=extra_fields, jit_model_args=jit_model_args) # sequential, vectorized

print('MCMC finished, collecting samples...')
samples = model.samples()
print(samples)
# file = f'/data/hz420-2/radial_migration_kernel/mock_sample/L_conditioned/Prior_distribution_{Nknots}knots.pkl'
file = f'/data/hz420-2/radial_migration_kernel/mock_sample/L_conditioned/results_errorfree_{Nknots}knots_L@10.pkl'
with open(file,'wb') as f:
    pickle.dump(samples, f)

ef = model.mcmc.get_extra_fields(group_by_chain=True,
                                )
num_steps = ef['num_steps']    # shape (n_chains, n_warmup+n_samples)
print("Avg leapfrog steps per sample:",
      num_steps.mean())
print("Total lnL/grad calls ≃", num_steps.sum())
