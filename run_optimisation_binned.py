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

import jax.scipy.optimize as jso

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
F_centre_for_sampling, F_scale_for_sampling = -0.5,  0.5

data_generated = generate_sample_for_MC_integration(data_grid, R_scale_for_sampling = R_scale_for_sampling, 
                                                    F_centre_for_sampling = F_centre_for_sampling, 
                                                    F_scale_for_sampling = F_scale_for_sampling, 
                                                    N_sample = int(1e4)) 

data_generated['weights'] = jnp.array(Nstars)


@jax.jit
def minimize_logL_numpyro(params, aux_params):

    Nknots = len(aux_params['aux_knots'])
    params_S = {'ln_Rdisk':params[:Nknots],
                'ln_sigmaLz':params[Nknots:2*Nknots],}

    time_start = time.perf_counter()
    # prior = -jnp.sum((params.reshape(2, Nknots)-aux_params['prior_mean'][:,np.newaxis])**2/2./aux_params['prior_std'][:,np.newaxis]**2)
    val = -jnp.sum(logL_numpyro(data_generated, params_S,
                                    **aux_params,))
    ln_prior = log_prior(params_S)
    time_end = time.perf_counter()
    #print(params_S, val)
    jax.debug.print('params: {params}, logL: {val}, log_prior:{z}, logP = {tot}, time: {t}', 
                    params=params_S, val=-val, z=ln_prior, tot=-val + ln_prior, t = time_end - time_start)
    return val - ln_prior



Nknots = 10
aux_params = {'R_scale_for_sampling':R_scale_for_sampling, 
              'F_scale_for_sampling':F_scale_for_sampling, 
              'F_centre_for_sampling':F_centre_for_sampling}  # Auxiliary parameters for the model
aux_params['aux_knots'] = generate_aux_knots(Nknots=Nknots, age_max=12.)

# Compute the ground truth parameters and log likelihood
age_pivot = np.array([0, 4, 8, 12, 20])
sigmaLz_pivot = np.array([10, 200, 400, 1200, 2000]) # sigma_Lz at 6 Gyr
sigmaLz_gt_interp = sp.interpolate.interp1d(age_pivot, sigmaLz_pivot, kind='cubic', fill_value='extrapolate', bounds_error=False)
Rd_gt = Rd_evolution_jump(np.linspace(0,12,100), Rdmax = 3.45, Rdmin = 1, tau_Rd = 7.0, delta_tau_Rd = 1.0)
Rd_gt_interp = sp.interpolate.interp1d(np.linspace(0,12,100), Rd_gt, kind='cubic', fill_value='extrapolate', bounds_error=False)
ln_Rdisk_knots = jnp.log(Rd_gt_interp(aux_params['aux_knots']))  # Convert Rd to ln(Rdisk)
ln_sigmaLz_knots = jnp.log(sigmaLz_gt_interp(aux_params['aux_knots']))  # Convert sigma_Lz to ln(sigma_Lz)

params_trial = {
    'ln_Rdisk': ln_Rdisk_knots,
    'ln_sigmaLz': ln_sigmaLz_knots,
}
log_prior1 = log_prior(params_trial)
logp1 = logL_numpyro(data_generated, params_trial, **aux_params)
print('GROUND TRUTH', 
      'params', params_trial, 'log_prior:', log_prior1, 'logp:', jnp.sum(logp1))

# Initial guess for the optimization
x0 = jnp.array(jnp.concatenate([
    jnp.ones(Nknots),  # ln_Rdisk knots
    6.0 * jnp.ones(Nknots)  # ln_sigmaLz knots
]))

print('Optimization begins')
min_results = jso.minimize(minimize_logL_numpyro, 
                           x0, 
                           args=(aux_params,),
                           method='BFGS',
                           tol = 1e-3,
                           )#options = {'maxiter':100}

res = min_results.x

file = f'/data/hz420-2/radial_migration_kernel/mock_sample/L_conditioned/minimization_result_{Nknots}knots_L@10.npy'
np.save(file, res)