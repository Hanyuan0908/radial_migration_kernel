import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Use CPU for JAX
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Use GPU for JAX

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

jax.config.update('jax_log_compiles', False)
np.random.seed(42)

Vc0 = 240
Nknots = 10
run_mock = True

path_to_dir = '/data/hz420-2/'
# path_to_dir = '/Users/hanyuan/Desktop/PhD_projects/'
# path_to_dir = '/home/yuxinyao/Desktop/'


if run_mock:
    # df_mockdata = pd.read_csv(path_to_dir+'radial_migration_kernel/mock_sample/L_conditioned/mock_data3.csv') # _errorfree_L@10
    # Rg_range = [9,11]
    # final_file = path_to_dir+f'radial_migration_kernel/mock_sample/L_conditioned/minimization_result_withMHgrad_{Nknots}knots4_binned.npy'#_L@10

    df_mockdata = pd.read_csv(path_to_dir+'radial_migration_kernel/mock_sample/with_bar/mock_data1.csv') # _errorfree_L@10
    Rg_range = [6,7]
    final_file = path_to_dir+f'radial_migration_kernel/mock_sample/with_bar/minimization_result_withMHgrad_{Nknots}knots1.npy'#_L@10

    df_mockdata = df_mockdata.sample(frac = 0.3, random_state=42)  # subsample for faster processing

else:
    df_realdata = pd.read_csv(path_to_dir+'catalogues/LAMOST/LAMOST_Gaia_subgiants_Xiangetal2024_kinematics.csv')
    F = df_realdata['FEH']
    L = df_realdata['Lz']
    age = df_realdata['AGE']
    sigma_F = df_realdata['E_FEH']
    sigma_L = df_realdata['E_Lz']
    e_age = df_realdata['E_AGE']
    logage_l, logage_h = np.log10(age-e_age), np.log10(age+e_age)
    logage = np.log10(age)
    sigma_logage = (logage_h - logage_l)/2

    L_range = [10*Vc0,11*Vc0]

    final_file = path_to_dir+f'radial_migration_kernel/results/minimization_result_{Nknots}knots_L@10to11.npy'#_L@10


F, logage, L = df_mockdata['MH'], df_mockdata['log_age'], df_mockdata['Lz']
sigma_F, sigma_logage, sigma_L = df_mockdata['sigma_MH'], df_mockdata['sigma_logage'], df_mockdata['sigma_Lz']


feh_range = [-1.5,0.5]
logage_range = np.log10([0.3,12])



# data_grid = binning_with_different_sigma(df_mockdata, 
#                                          MH_sigma_level=[0,0.02,0.05,0.1],
#                                          logage_sigma_level=[0,0.02,0.05,0.1],
#                                          feh_range=feh_range,
#                                          logage_range=logage_range,
#                                          Rg_range = Rg_range,
#                                          Vc0 = 240.,)

#=====================================================================================

data_grid = {
    'MH': jnp.array(F),
    'log_age': jnp.array(logage),
    'Lz': jnp.array(L),
    'sigma_MH': jnp.array(sigma_F),
    'sigma_logage': jnp.array(sigma_logage),
    'sigma_Lz': jnp.array(sigma_L),
    'Nstars': jnp.ones_like(F),  # Assuming equal weights for simplicity
}
#=====================================================================================

print('Number of grid points:', len(data_grid['MH']))
print('Number of total stars:', jnp.sum(data_grid['Nstars']))
N_sample = int(2e3)
# F_centre_for_sampling, F_scale_for_sampling = -0.5,  0.5
R_scale_for_sampling = 4
R_scale_at_0, R_scale_at_12 = 4, 1
F_centre_at_0, F_centre_at_12 = -0.1, -0.7
F_scale_at_0, F_scale_at_12 = 0.1, 0.7

data_generated = generate_sample_for_MC_integration_withprob_samenumdenom(data_grid,
                                    R_scale_at_0 = R_scale_at_0, R_scale_at_12 = R_scale_at_12,
                                    F_centre_at_0=F_centre_at_0, F_centre_at_12=F_centre_at_12,
                                    F_scale_at_0=F_scale_at_0, F_scale_at_12=F_scale_at_12,
                                    N_sample = N_sample)

# data_generated = generate_sample_for_MC_integration_withprob(data_grid, R_scale_for_sampling = R_scale_for_sampling, 
#                                                     F_centre_at_0=F_centre_at_0, F_centre_at_12=F_centre_at_12,
#                                                     F_scale_at_0=F_scale_at_0, F_scale_at_12=F_scale_at_12,
#                                                     N_sample = N_sample) 

# data_generated['weights'] = jnp.ones(len(F))  # Weights for the generated sample
data_generated['weights'] = data_grid['Nstars']  # Use the original weights from the data grid
shape = data_generated['age_sample'].shape
age = data_generated['age_sample'].reshape(-1)
data_generated['MH_max_sample'] = MH_evolution_Lu24(age, 0).reshape(shape)

@jax.jit
def minimize_logL_numpyro(params, aux_params):

    Nknots = len(aux_params['aux_knots'])
    params_S = {'ln_Rdisk':params[:Nknots],
                'ln_sigmaLz':params[Nknots:2*Nknots],
                'ln_MH_grad':params[2*Nknots:3*Nknots],
                }

    time_start = time.perf_counter()
    # prior = -jnp.sum((params.reshape(2, Nknots)-aux_params['prior_mean'][:,np.newaxis])**2/2./aux_params['prior_std'][:,np.newaxis]**2)
    val = -jnp.sum(logL_numpyro_withMHmodel3(data_generated, params_S,
                                    **aux_params,))
    lnPrior1 = lnRdisk_prior_normal(params_S)
    lnPrior2 = lnSigmaLz_prior_normal(params_S)
    # lnPrior3 = MH_max_param_prior_normal(params_S)
    lnPrior4 = ln_MH_grad_prior_normal(params_S)
    # lnPrior5 = ln_MH_grad_prior_uniform(params_S)
    lnPrior_smooth = smoothing_prior_withMHmodel2(params_S)
    ln_prior = lnPrior1 + lnPrior2 + lnPrior4 + lnPrior_smooth

    time_end = time.perf_counter()
    #print(params_S, val)
    jax.debug.print('params: {params}, logL: {val}, log_prior:{z}, logP = {tot}, time: {t}', 
                    params=params_S, val=-val, z=ln_prior, tot=-val + ln_prior, t = time_end - time_start)
    return val - ln_prior



aux_params = {'ln_MH_grad_0': -2.66, 'tol': 5e-2, 'Vc0':240.}  # Auxiliary parameters for the model
aux_params['aux_knots'] = generate_aux_knots(Nknots=Nknots, age_max=12.)

# Compute the ground truth parameters and log likelihood
# age_pivot = np.array([0, 4, 8, 12, 20])
# sigmaLz_pivot = np.array([10, 200, 400, 1200, 2000]) # sigma_Lz at 6 Gyr
# sigmaLz_gt_interp = sp.interpolate.interp1d(age_pivot, sigmaLz_pivot, kind='cubic', fill_value='extrapolate', bounds_error=False)
# Rd_gt = Rd_evolution_jump(np.linspace(0,12,100), Rdmax = 3.45, Rdmin = 1, tau_Rd = 7.0, delta_tau_Rd = 1.0)
# Rd_gt_interp = sp.interpolate.interp1d(np.linspace(0,12,100), Rd_gt, kind='cubic', fill_value='extrapolate', bounds_error=False)

age_pivot = np.array([0, 4, 8, 12, 20])
sigmaLz_pivot = np.array([0, 300, 600, 1500, 2000]) # sigma_Lz at 6 Gyr
sigmaLz_gt_interp = sp.interpolate.interp1d(age_pivot, sigmaLz_pivot, kind='cubic', fill_value='extrapolate', bounds_error=False)
Rd_gt = Rd_evolution_jump(np.linspace(0,15,100), Rdmax = 2.72, Rdmin = 1, tau_Rd = 6.0, delta_tau_Rd = 3.0)
Rd_gt_interp = sp.interpolate.interp1d(np.linspace(0,15,100), Rd_gt, kind='cubic', fill_value='extrapolate', bounds_error=False)

# MH_at_8_gt = MH_evolution_Lu24(np.linspace(0,20,100), 8 * Vc0,)
# MH_at_8_gt_interp = sp.interpolate.interp1d(np.linspace(0,20,100), MH_at_8_gt, kind='cubic', fill_value='extrapolate', bounds_error=False)
MH_grad_gt = (MH_evolution_Lu24(np.linspace(0,20,100), 9 * Vc0) - MH_evolution_Lu24(np.linspace(0,20,100), 8 * Vc0))
ln_MH_grad_gt = np.log(-MH_grad_gt)
ln_MH_grad_gt_interp = sp.interpolate.interp1d(np.linspace(0,20,100), ln_MH_grad_gt, kind='cubic', fill_value='extrapolate', bounds_error=False)

ln_Rdisk_knots = jnp.log(Rd_gt_interp(aux_params['aux_knots']))  # Convert Rd to ln(Rdisk)
ln_sigmaLz_knots = jnp.log(sigmaLz_gt_interp(aux_params['aux_knots']) / aux_params['aux_knots'])  # Convert sigma_Lz to ln(sigma_Lz)
ln_sigmaLz_knots = ln_sigmaLz_knots.at[0].set(4.)  # Ensure the first knot is set to the first pivot value

# MH_at_8_knots = MH_at_8_gt_interp(aux_params['aux_knots'])
ln_MH_grad_knots = ln_MH_grad_gt_interp(aux_params['aux_knots'])

MH_at_0_gt = MH_evolution_Lu24(np.linspace(0,20,100), 0 * Vc0,)
MH_at_0_gt_interp = sp.interpolate.interp1d(np.linspace(0,20,100), MH_at_0_gt, kind='cubic', fill_value='extrapolate', bounds_error=False)
Z0, m1, m2 = 0.6, 0.02, 0.2
t_s, t_scale = 10.0, 1
t = np.linspace(0, 12, 400)
Z = MH_max_func(t, Z0, t_s, t_scale, m1, m2)
import scipy.optimize
fitted_params, _= scipy.optimize.curve_fit(MH_max_func, t, MH_at_0_gt_interp(t), p0=[Z0, t_s, t_scale, m1, m2], sigma = 0.05)
print('Fitted parameters:', fitted_params)
MH_max_params_gt = np.log(fitted_params[1:])

params_trial = {
    'ln_Rdisk': ln_Rdisk_knots,
    'ln_sigmaLz': ln_sigmaLz_knots,
    'ln_MH_grad': ln_MH_grad_knots,
}
log_prior1 = log_prior(params_trial)
logp1 = logL_numpyro_withMHmodel3(data_generated, params_trial, **aux_params)
print('GROUND TRUTH', 
      'params', params_trial, 'log_prior:', log_prior1, 'logp:', jnp.sum(logp1))

# Initial guess for the optimization

# x0 = jnp.array(jnp.concatenate([
#     jnp.array(np.random.uniform(0,1., Nknots)),  # ln_Rdisk knots
#     jnp.array(np.random.uniform(4,5, Nknots)),  # ln_sigmaLz knots
#     jnp.log(np.random.uniform(0.05, 0.10, Nknots)),
# ]))

minimiser_results = np.load(final_file)
ln_Rdisk_minimiser = jnp.array(minimiser_results[:Nknots])
ln_sigmaLz_minimiser = jnp.array(minimiser_results[Nknots:2*Nknots])
ln_MH_grad_minimiser = jnp.array(minimiser_results[2*Nknots:3*Nknots])
MH_max_param_minimiser = jnp.array(minimiser_results[3*Nknots:])
# MH_at_8_minimiser = MH_at_8_minimiser.at[0].set(0.064)  # Ensure the first knot is set to zero
ln_MH_grad_minimiser = ln_MH_grad_minimiser.at[0].set(-2.6)  # Ensure the first knot is set to a reasonable value
ln_MH_grad_minimiser = ln_MH_grad_minimiser.at[-1].set(-2) 
x0 = jnp.array(jnp.concatenate([
                jnp.array(ln_Rdisk_minimiser),
                jnp.array(ln_sigmaLz_minimiser),
                jnp.array(ln_MH_grad_minimiser),
                jnp.array(MH_max_param_minimiser)
                ]))

print('Optimization begins')
min_results = jso.minimize(minimize_logL_numpyro, 
                           x0, 
                           args=(aux_params,),
                           method='BFGS',
                           tol = 1e-4,
                           )#options = {'maxiter':100}

res = min_results.x

file = final_file #f'/data/hz420-2/radial_migration_kernel/mock_sample/L_conditioned/minimization_result_{Nknots}knots2.npy'#_L@10
np.save(file, res)