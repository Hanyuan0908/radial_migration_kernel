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
import time

Vc0 = 240.  # km/s, circular velocity at the solar radius
# df_mockdata = pd.read_csv('/data/hz420-2/radial_migration_kernel/mock_sample/L_conditioned/mock_data.csv')
# df_mockdata = df_mockdata.sample(frac=0.1, random_state=42)
# data_mock = {
#     'MH': jnp.array(df_mockdata['MH'].to_numpy()),
#     'logage': jnp.array(df_mockdata['log_age'].to_numpy()),
#     'Lz': jnp.array(df_mockdata['Lz'].to_numpy()),
#     'sigma_MH': jnp.array(df_mockdata['sigma_MH'].to_numpy()),
#     'sigma_logage': jnp.array(df_mockdata['sigma_logage'].to_numpy()),
#     'sigma_Lz': jnp.array(df_mockdata['sigma_Lz'].to_numpy()),
# }

# def minimize_logL_numpyro(params, aux_params):

#     Nknots = len(aux_params['aux_knots'])
#     params_S = {'ln_Rdisk':params[:Nknots],
#                 'ln_sigmaLz':params[Nknots:2*Nknots],}

#     # prior = -jnp.sum((params.reshape(2, Nknots)-aux_params['prior_mean'][:,np.newaxis])**2/2./aux_params['prior_std'][:,np.newaxis]**2)
#     val = -logL_numpyro(data_mock, params_S,
#                                     **aux_params,)
#     # print(params_S, val)
#     return val# - prior

# age_pivot = np.array([0, 4, 8, 12])
# sigmaLz_pivot = np.array([10, 200, 400, 1200]) # sigma_Lz at 6 Gyr
# sigmaLz_gt_interp = sp.interpolate.interp1d(age_pivot, sigmaLz_pivot, kind='cubic', fill_value='extrapolate', bounds_error=False)
# Rd_gt = Rd_evolution_jump(np.linspace(0,12,100), Rdmax = 3.45, Rdmin = 1, tau_Rd = 7.0, delta_tau_Rd = 1.0)
# Rd_gt_interp = sp.interpolate.interp1d(np.linspace(0,12,100), Rd_gt, kind='cubic', fill_value='extrapolate', bounds_error=False)

# Nknots = 5
# aux_knots = jnp.linspace(0.,12.,Nknots)
# aux_params = {}  # Auxiliary parameters for the model
# aux_params['aux_knots'] = jnp.linspace(0.,12.,Nknots)

# ln_Rdisk_knots = jnp.log(Rd_gt_interp(aux_knots))  # Convert Rd to ln(Rdisk)
# ln_sigmaLz_knots = jnp.log(sigmaLz_gt_interp(aux_knots))  # Convert sigma_Lz to ln(sigma_Lz)
# # x0 = jnp.concatenate([ln_Rdisk_knots, ln_sigmaLz_knots])
# # print(x0)
# # print(minimize_logL_numpyro(x0, aux_params))
# # x0 = jnp.array([1.,1.,1.,1.,1.,6.,6.,6.,6.,6.])
# # print(x0)
# # print(minimize_logL_numpyro(x0, aux_params))

# params_trial = {
#     'ln_Rdisk': ln_Rdisk_knots,
#     'ln_sigmaLz': ln_sigmaLz_knots,
# }
# start = time.perf_counter()
# log_prior1 = log_prior(params_trial)
# logp1 = logL_numpyro(data_mock, params_trial, **aux_params)
# # print('params', params_trial)
# end = time.perf_counter()
# print(f"time taken: {end - start:.4f} s")


# params_trial = {
#     'ln_Rdisk': jnp.array([1.,1.,1.,1.,1.]),
#     'ln_sigmaLz': jnp.array([6.,6.,6.,6.,6.]),
# }
# start = time.perf_counter()
# log_prior2 = log_prior(params_trial)
# logp2 = logL_numpyro(data_mock, params_trial, **aux_params)
# # print('params', params_trial)
# end = time.perf_counter()
# print(f"time taken: {end - start:.4f} s")


# params_trial = {
#     'ln_Rdisk': jnp.array([1.,1.,1.,1.,1.]),
#     'ln_sigmaLz': jnp.array([4.,5.,6.,6.5,7.]),
# }
# start = time.perf_counter()
# log_prior3 = log_prior(params_trial)
# logp3 = logL_numpyro(data_mock, params_trial, **aux_params)
# # print('params', params_trial)
# end = time.perf_counter()
# print(f"time taken: {end - start:.4f} s")

# print('logp1:', logp1, 'logp2:', logp2, 'logp3:', logp3)
# print('log_prior1:', log_prior1, 'log_prior2:', log_prior2, 'log_prior3:', log_prior3)

# params_trial = {
#     'ln_Rdisk': jnp.array([10.,1.,1.,1.,1.]),
#     'ln_sigmaLz': jnp.array([6.,10.,6.,6.,6.]),
# }
# log_prior_test = log_prior(params_trial)
# print('log_prior_test:', log_prior_test)

L_val = 10 * Vc0
Z_grid = jnp.arange(-1.5, 0.8, 0.05)
age_grid = jnp.arange(0.1, 12, 0.2)
log_age_grid = jnp.log10(age_grid)

Z_grid, log_age_grid = jnp.meshgrid(Z_grid, log_age_grid)
Z_grid = Z_grid.flatten()
log_age_grid = log_age_grid.flatten()

print('number of points in the grid:', len(Z_grid))

L_grid = jnp.ones_like(Z_grid) * L_val
sigmaZ_grid = jnp.ones_like(Z_grid) * 0.05 # Assuming a constant sigma_F for simplicity
sigma_logage_grid = jnp.ones_like(log_age_grid) * 0.  # Assuming a constant sigma_logage for simplicity
sigmaLz_grid = jnp.ones_like(Z_grid) * 30  # Assuming a constant sigma_Lz for simplicity

# Generate samples
N_star = len(Z_grid)  # Number of stars in the grid
N_sample = int(1e3)
R_scale_for_sampling = 4
F_centre_for_sampling, F_scale_for_sampling = -0.5,  0.5
jax_random_key1 = jax.random.PRNGKey(42)
jax_random_key2 = jax.random.PRNGKey(10086)
jax_random_key3 = jax.random.PRNGKey(10010)
jax_random_key4 = jax.random.PRNGKey(999)
jax_random_key5 = jax.random.PRNGKey(2025)
jax_random_key6 = jax.random.PRNGKey(124)
jax_random_key7 = jax.random.PRNGKey(456)
jax_random_key8 = jax.random.PRNGKey(789)


F_stack = jnp.repeat(Z_grid[None, :], N_sample, axis=0)
sigmaF_stack = jnp.repeat(sigmaZ_grid[None, :], N_sample, axis=0)
L_stack = jnp.repeat(L_grid[None, :], N_sample, axis=0)
sigmaL_stack = jnp.repeat(sigmaLz_grid[None, :], N_sample, axis=0)
logage_stack = jnp.repeat(log_age_grid[None, :], N_sample, axis=0)
sigmalogage_stack = jnp.repeat(sigma_logage_grid[None, :], N_sample, axis=0)

# L0_sample1 = jax.random.exponential(jax_random_key1, shape=(N_sample,N_star)) * (R_scale_for_sampling * Vc0) # * Lcentre
# L0_sample2 = jax.random.exponential(jax_random_key8, shape=(N_sample,N_star)) * (R_scale_for_sampling * Vc0) # * Lcentre
rng = np.random.default_rng(1234)
L0_sample1 = jnp.array(sample_x_exp(rng, R_scale_for_sampling * Vc0, size=(N_sample,N_star)))
rng = np.random.default_rng(5678)
L0_sample2 = jnp.array(sample_x_exp(rng, R_scale_for_sampling * Vc0, size=(N_sample,N_star)))

F_sample1 = jax.random.normal(jax_random_key2, shape=(N_sample,N_star)) * sigmaF_stack + F_stack
logage_sample1 = jax.random.normal(jax_random_key3, shape=(N_sample,N_star)) * sigmalogage_stack + logage_stack
L_sample1 = jax.random.normal(jax_random_key4, shape=(N_sample,N_star)) * sigmaL_stack + L_stack
F_sample2 = jax.random.normal(jax_random_key5, shape=(N_sample,N_star)) * F_scale_for_sampling + F_centre_for_sampling  # Assuming a standard normal distribution for F
L_sample2 = jax.random.normal(jax_random_key6, shape=(N_sample,N_star)) * sigmaL_stack + L_stack
logage_sample2 = jax.random.normal(jax_random_key7, shape=(N_sample,N_star)) * sigmalogage_stack + logage_stack

age_sample1 = 10**(logage_sample1)
age_sample1 = jnp.where(age_sample1 > 15, 15, age_sample1)
age_sample2 = 10**(logage_sample2)
age_sample2 = jnp.where(age_sample2 > 15, 15, age_sample2)

data_test = {
    'L0_sample_num': L0_sample1,
    'L0_sample_denom': L0_sample2,
    'F_sample_num': F_sample1,
    'F_sample_denom': F_sample2,
    'age_sample_num': age_sample1,
    'age_sample_denom': age_sample2,
    'L_sample_num': L_sample1,
    'L_sample_denom': L_sample2,
    'weights': jnp.ones(N_star)
}


age_pivot = np.array([0, 4, 8, 12, 20])
sigmaLz_pivot = np.array([10, 200, 400, 1200, 2000]) # sigma_Lz at 6 Gyr
sigmaLz_gt_interp = sp.interpolate.interp1d(age_pivot, sigmaLz_pivot, kind='cubic', fill_value='extrapolate', bounds_error=False)
Rd_gt = Rd_evolution_jump(np.linspace(0,15,100), Rdmax = 3.45, Rdmin = 1, tau_Rd = 7.0, delta_tau_Rd = 1.0)
Rd_gt_interp = sp.interpolate.interp1d(np.linspace(0,15,100), Rd_gt, kind='cubic', fill_value='extrapolate', bounds_error=False)

Nknots = 10
# aux_knots = jnp.linspace(0.,15.,Nknots)
aux_params = {'R_scale_for_sampling':R_scale_for_sampling, 
              'F_scale_for_sampling':F_scale_for_sampling, 
              'F_centre_for_sampling':F_centre_for_sampling}  # Auxiliary parameters for the model
aux_params['aux_knots'] = generate_aux_knots(Nknots=Nknots, age_max=12.)#jnp.linspace(0.,15.,Nknots)

ln_Rdisk_knots = jnp.log(Rd_gt_interp(aux_params['aux_knots']))  # Convert Rd to ln(Rdisk)
ln_sigmaLz_knots = jnp.log(sigmaLz_gt_interp(aux_params['aux_knots']))  # Convert sigma_Lz to ln(sigma_Lz)
# params_trial = {
#     'ln_Rdisk': ln_Rdisk_knots,
#     'ln_sigmaLz': ln_sigmaLz_knots,
# }

# params_trial = {'ln_Rdisk': jnp.array([ 1.0129285 ,  1.1170565 ,  1.1829929 ,  1.0032728 , -0.00879301,
#        -0.89356416,  0.7450388 ,  1.0250543 ]), 
#        'ln_sigmaLz': jnp.array([5.6591077, 3.8715577, 3.332939 , 4.028008 , 6.5052104, 6.417953 ,
#        6.1114144, 5.9913044])
#        }


# params_trial= {'ln_Rdisk': jnp.array([ 1.0166726 ,  1.0310085 ,  1.3933964 ,  0.7371091 , -0.90100026,
#         0.2041145 ,  0.08324957,  1.1941382 ]), 'ln_sigmaLz': jnp.array([4.0198526, 2.902585 , 5.4031043, 5.091751 , 5.7446647, 6.606856 ,
#        7.7502027, 5.8808165])}

file_name = f'/data/hz420-2/radial_migration_kernel/mock_sample/L_conditioned/minimization_result_{Nknots}knots_L@10.npy'
minimiser_results = np.load(file_name)
params_trial = {
    'ln_Rdisk': jnp.array(minimiser_results[:Nknots]),
    'ln_sigmaLz': jnp.array(minimiser_results[Nknots:2*Nknots]),
}

time_start = time.perf_counter()
logL_numpyro_val = logL_numpyro(data_test, params_trial, **aux_params)
time_end = time.perf_counter()
print(f"time taken for logL_numpyro: {time_end - time_start:.4f} s")

fig, ax = plt.subplots(1, 4, figsize=(20, 4), gridspec_kw={'width_ratios': [1, 1, 1, 2], 'wspace': 0.35})
Rdiskinterp = InterpolatedUnivariateSpline(aux_params['aux_knots'], np.exp(params_trial['ln_Rdisk']), k=3)
ax[0].plot(np.linspace(0,12,100), Rdiskinterp(np.linspace(0,12,100)), ls = '-', label='Rdisk knots')
ax[0].plot(aux_params['aux_knots'], np.exp(params_trial['ln_Rdisk']), color = 'k', marker='o', ls='None', label='Rdisk knots')
ax[0].set_xlabel('Age (Gyr)')
ax[0].set_xlim(0,12)
ax[0].set_ylabel('Rdisk (kpc)')
sigmaLzinterp = InterpolatedUnivariateSpline(aux_params['aux_knots'], np.exp(params_trial['ln_sigmaLz']), k=3)
ax[1].plot(np.linspace(0,12,100), sigmaLzinterp(np.linspace(0,12,100)), ls = '-', label='sigmaLz knots')
ax[1].plot(aux_params['aux_knots'], np.exp(params_trial['ln_sigmaLz']), color = 'k', marker='o', ls='None', label='sigmaLz knots')
ax[1].set_xlabel('Age (Gyr)')
ax[1].set_xlim(0,12)
ax[1].set_ylabel(r'$\sigma_{Lz}$ (kpc km/s)')

for i in range (0, 15):
    ax[2].plot(np.linspace(0,12,100), MH_evolution_Lu24(np.linspace(0,12,100), i*Vc0), label=f'MH={i/10:.1f}')
ax[2].set_xlabel('Age (Gyr)')
ax[2].set_ylabel('Metallicity (Z)')

cb = ax[3].scatter(10**(log_age_grid), Z_grid, c=np.exp(logL_numpyro_val), cmap='jet', 
                vmin = np.percentile(np.exp(logL_numpyro_val), 5), vmax = np.percentile(np.exp(logL_numpyro_val), 99))
ax[3].set_xlabel('Age (Gyr)')
ax[3].set_ylabel('Metallicity')
fig.colorbar(cb, ax=ax[3], label='exp(Log Likelihood)')

logage_unique = jnp.unique(log_age_grid)
ls = []
for i in range(len(logage_unique)):
    mask = log_age_grid == logage_unique[i]
    logL_numpyro_val_i = logL_numpyro_val[mask]
    ls.append(float(jsp.special.logsumexp(logL_numpyro_val_i)))

print(ls)

print_ground_truth = True
if print_ground_truth:
    Rdiskinterp = InterpolatedUnivariateSpline(aux_params['aux_knots'], np.exp(ln_Rdisk_knots), k=3)
    ax[0].plot(np.linspace(0,12,100), Rd_gt_interp(np.linspace(0,12,100)), label='Rdisk knots', color='red', ls='--')
    sigmaLzinterp = InterpolatedUnivariateSpline(aux_params['aux_knots'], np.exp(ln_sigmaLz_knots), k=3)
    ax[1].plot(np.linspace(0,12,100), sigmaLz_gt_interp(np.linspace(0,12,100)), label='sigmaLz knots', color='red', ls='--')


fig.savefig('/data/hz420-2/radial_migration_kernel/logL_numpyro_test.png', dpi=300, bbox_inches='tight')
plt.show()


# #%%
# logL_numpyro_val = logP_F_given_tau_L(Z_grid, L_grid, 10**log_age_grid, params_trial, aux_knots = aux_params['aux_knots'])

# fig, ax = plt.subplots(1, 4, figsize=(20, 3.5), gridspec_kw={'width_ratios': [1, 1, 1, 2], 'wspace': 0.35})
# Rdiskinterp = InterpolatedUnivariateSpline(aux_knots, np.exp(ln_Rdisk_knots), k=3)
# ax[0].plot(np.linspace(0,12,100), Rdiskinterp(np.linspace(0,12,100)), 'o-', label='Rdisk knots')
# ax[0].set_xlabel('Age (Gyr)')
# ax[0].set_ylabel('Rdisk (kpc)')
# sigmaLzinterp = InterpolatedUnivariateSpline(aux_knots, np.exp(ln_sigmaLz_knots), k=3)
# ax[1].plot(np.linspace(0,12,100), sigmaLzinterp(np.linspace(0,12,100)), 'o-', label='sigmaLz knots')
# ax[1].set_xlabel('Age (Gyr)')
# ax[1].set_ylabel('Rdisk (kpc)')

# for i in range (0, 15):
#     ax[2].plot(np.linspace(0,12,100), MH_evolution_Lu24(np.linspace(0,12,100), i*Vc0), label=f'MH={i/10:.1f}')
# ax[2].set_xlabel('Age (Gyr)')
# ax[2].set_ylabel('Metallicity (Z)')

# cb = ax[3].scatter(10**(log_age_grid), Z_grid, c=np.exp(logL_numpyro_val), cmap='jet', 
#                 vmin = np.percentile(np.exp(logL_numpyro_val), 10), vmax = np.percentile(np.exp(logL_numpyro_val), 95))
# ax[3].set_xlabel('Age (Gyr)')
# ax[3].set_ylabel('Metallicity')
# fig.colorbar(cb, ax=ax[3], label='exp(Log Likelihood)')
# fig.savefig('/data/hz420-2/radial_migration_kernel/logL_numpyro_test2.png', dpi=300)
# plt.show()

# logage_unique = jnp.unique(log_age_grid)
# ls = []
# for i in range(len(logage_unique)):
#     mask = log_age_grid == logage_unique[i]
#     logL_numpyro_val_i = logL_numpyro_val[mask]
#     ls.append(jsp.special.logsumexp(logL_numpyro_val_i))

# print(ls)
