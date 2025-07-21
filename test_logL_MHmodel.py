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
Nknots = 10
check_mock = True
path_to_dir = '/data/hz420-2/'  # Path to the data directory
# path_to_dir = '/Users/hanyuan/Desktop/PhD_projects/'

if check_mock:
    read_file = path_to_dir+f'radial_migration_kernel/mock_sample/L_conditioned/minimization_result_{Nknots}knots8.npy'
    figure_name = path_to_dir+f'radial_migration_kernel/mock_sample/L_conditioned/optimisation_results_{Nknots}knots_4.png'
else:
    read_file = path_to_dir+f'radial_migration_kernel/results/minimization_result_{Nknots}knots_L@10to11.npy'
    figure_name = path_to_dir+f'radial_migration_kernel/results/optimisation_results_{Nknots}knots_L@10to11.png'



L_val = 10 * Vc0
Z_grid = jnp.arange(-1.5, 0.8, 0.05)
age_grid = jnp.arange(0.1, 12, 0.2)
log_age_grid = jnp.log10(age_grid)

Z_grid, log_age_grid = jnp.meshgrid(Z_grid, log_age_grid)
Z_grid = Z_grid.flatten()
log_age_grid = log_age_grid.flatten()

print('number of points in the grid:', len(Z_grid))

L_grid = jnp.ones_like(Z_grid) * L_val
sigmaZ_grid = jnp.ones_like(Z_grid) * 0.02 # Assuming a constant sigma_F for simplicity
sigma_logage_grid = jnp.ones_like(log_age_grid) * 0.03  # Assuming a constant sigma_logage for simplicity
sigmaLz_grid = jnp.ones_like(Z_grid) * 30  # Assuming a constant sigma_Lz for simplicity

data_grid = {
    'MH': jnp.array(Z_grid),
    'log_age': jnp.array(log_age_grid),
    'Lz': jnp.array(L_grid),
    'sigma_MH': jnp.array(sigmaZ_grid) ,
    'sigma_logage': jnp.array(sigma_logage_grid) * 1.,
    'sigma_Lz': jnp.array(sigmaLz_grid),
}

# Generate samples
N_star = len(Z_grid)  # Number of stars in the grid
N_sample = int(1e3)
# F_centre_for_sampling, F_scale_for_sampling = -0.5,  0.5
R_scale_at_0, R_scale_at_12 = 4, 1
F_centre_at_0, F_centre_at_12 = -0.1, -0.8
F_scale_at_0, F_scale_at_12 = 0.1, 0.7

data_test = generate_sample_for_MC_integration_withprob_samenumdenom(data_grid,
                                    R_scale_at_0 = R_scale_at_0, R_scale_at_12 = R_scale_at_12,
                                    F_centre_at_0=F_centre_at_0, F_centre_at_12=F_centre_at_12,
                                    F_scale_at_0=F_scale_at_0, F_scale_at_12=F_scale_at_12,
                                    N_sample = N_sample)
# data_test = generate_sample_for_MC_integration_withprob(data_grid,
#                                     R_scale_for_sampling= 4,
#                                     F_scale_at_0=F_scale_at_0, F_scale_at_12=F_scale_at_12,
#                                     N_sample = N_sample)

data_test['weights'] = jnp.ones(N_star)


age_pivot = np.array([0, 4, 8, 12, 20])
sigmaLz_pivot = np.array([10, 200, 400, 1200, 2000]) # sigma_Lz at 6 Gyr
sigmaLz_gt_interp = sp.interpolate.interp1d(age_pivot, sigmaLz_pivot, kind='cubic', fill_value='extrapolate', bounds_error=False)
Rd_gt = Rd_evolution_jump(np.linspace(0,15,100), Rdmax = 3.45, Rdmin = 1, tau_Rd = 7.0, delta_tau_Rd = 1.0)
Rd_gt_interp = sp.interpolate.interp1d(np.linspace(0,15,100), Rd_gt, kind='cubic', fill_value='extrapolate', bounds_error=False)

# aux_knots = jnp.linspace(0.,15.,Nknots)
aux_params = {}  # Auxiliary parameters for the model
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

print(read_file)
file_name = read_file
minimiser_results = np.load(file_name)
params_trial = {
    'ln_Rdisk': jnp.array(minimiser_results[:Nknots]),
    # 'ln_Rdisk': jnp.array(np.flip(np.linspace(-0.2, 1.2, Nknots))),
    'ln_sigmaLz': jnp.array(minimiser_results[Nknots:2*Nknots]),
}

time_start = time.perf_counter()
logL_numpyro_val = logL_numpyro4(data_test, params_trial, **aux_params)
time_end = time.perf_counter()
print(f"time taken for logL_numpyro: {time_end - time_start:.4f} s")

fig, ax = plt.subplots(1, 4, figsize=(20, 4), gridspec_kw={'width_ratios': [1, 1, 1, 2], 'wspace': 0.35})
Rdiskinterp = InterpolatedUnivariateSpline(aux_params['aux_knots'], np.exp(params_trial['ln_Rdisk']), k=3)
ax[0].plot(np.linspace(0,12,100), Rdiskinterp(np.linspace(0,12,100)), ls = '-', label='Minimiser results', lw = 3)
ax[0].plot(aux_params['aux_knots'], np.exp(params_trial['ln_Rdisk']), color = 'k', marker='o', ls='None', label='knots')
ax[0].set_xlabel('Age (Gyr)')
ax[0].set_xlim(12,0)
ax[0].set_ylabel('Rdisk (kpc)')
sigmaLzinterp = InterpolatedUnivariateSpline(aux_params['aux_knots'], np.exp(params_trial['ln_sigmaLz']) * aux_params['aux_knots'], k=3)
ax[1].plot(np.linspace(0,12,100), sigmaLzinterp(np.linspace(0,12,100)), ls = '-', label='Minimiser results', lw = 3)
ax[1].plot(aux_params['aux_knots'], np.exp(params_trial['ln_sigmaLz']) * aux_params['aux_knots'], color = 'k', marker='o', ls='None', label='knots')
ax[1].set_xlabel('Age (Gyr)')
ax[1].set_xlim(12,0)
ax[1].set_ylim(0, 1500)
ax[1].set_ylabel(r'$\sigma_{Lz}$ (kpc km/s)')

for i in range (0, 15):
    ax[2].plot(np.linspace(0,12,100), MH_evolution_Lu24(np.linspace(0,12,100), i*Vc0), label=f'MH={i/10:.1f}')
ax[2].set_xlabel('Age (Gyr)')
ax[2].set_ylabel('Metallicity (Z)')
ax[2].set_xlim(12,0)

cb = ax[3].scatter(10**(log_age_grid), Z_grid, c=np.exp(logL_numpyro_val), cmap='jet', 
                vmin = np.percentile(np.exp(logL_numpyro_val), 5), vmax = np.percentile(np.exp(logL_numpyro_val), 99))
ax[3].set_xlabel('Age (Gyr)')
ax[3].set_ylabel('Metallicity')
ax[3].set_xlim(12,0)
fig.colorbar(cb, ax=ax[3], label='exp(Log Likelihood)')

logage_unique = jnp.unique(log_age_grid)
ls = []
for i in range(len(logage_unique)):
    mask = log_age_grid == logage_unique[i]
    logL_numpyro_val_i = logL_numpyro_val[mask]
    ls.append(float(jsp.special.logsumexp(logL_numpyro_val_i)))

print(ls)

print_ground_truth = check_mock
if print_ground_truth:
    Rdiskinterp = InterpolatedUnivariateSpline(aux_params['aux_knots'], np.exp(ln_Rdisk_knots), k=3)
    ax[0].plot(np.linspace(0,12,100), Rd_gt_interp(np.linspace(0,12,100)), label='Ground truth', color='red', ls='--', lw = 2)
    sigmaLzinterp = InterpolatedUnivariateSpline(aux_params['aux_knots'], np.exp(ln_sigmaLz_knots), k=3)
    ax[1].plot(np.linspace(0,12,100), sigmaLz_gt_interp(np.linspace(0,12,100)), label='Ground truth', color='red', ls='--', lw = 2)

ax[0].legend()
ax[1].legend()

figure_name = figure_name
# fig.savefig(figure_name, dpi=300, bbox_inches='tight')
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
