from numpyro_model import *
from model_new import *

import numpy as np
import scipy as sp
import pandas as pd
import jax
# jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision
import jax.numpy as jnp
import numpyro
import jax.scipy as jsp
import pickle

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

R_scale_for_sampling = 6
F_centre_for_sampling, F_scale_for_sampling = -0.3,  1

data_generated = generate_sample_for_MC_integration(data_grid, R_scale_for_sampling = R_scale_for_sampling, 
                                                    F_centre_for_sampling = F_centre_for_sampling, 
                                                    F_scale_for_sampling = F_scale_for_sampling, 
                                                    N_sample = int(1e3)) 

data_generated['weights'] = jnp.array(Nstars)


# file = '/data/hz420-2/radial_migration_kernel/mock_sample/L_conditioned/Prior_distribution.pkl'
# with open(file, 'rb') as f:
#     prior_distribution = pickle.load(f)

# ln_Rdisk = np.array(prior_distribution['ln_Rdisk'])
# ln_sigmaLz = np.array(prior_distribution['ln_sigmaLz'])


# N_sample = ln_Rdisk.shape[0]
# N_knots = ln_Rdisk.shape[1]
# logL_ls = []

# aux_knots = jnp.linspace(0.,15.,N_knots)
# aux_params = {'R_scale_for_sampling':R_scale_for_sampling, 
#             'F_scale_for_sampling':F_scale_for_sampling, 
#             'F_centre_for_sampling':F_centre_for_sampling}  # Auxiliary parameters for the model
# aux_params['aux_knots'] = jnp.linspace(0.,15.,N_knots)

# for i in tqdm(range(N_sample)):
#     # aux_params['aux_knots'] = jnp.append(aux_params['aux_knots'], 20.)  # Add an extra knot at 20 Gyr

#     # params = {
#     #     'ln_Rdisk': jnp.array(jnp.append(ln_Rdisk[i], jnp.log(1))),
#     #     'ln_sigmaLz': jnp.array(jnp.append(ln_sigmaLz[i], jnp.log(2000))),
#     # }
#     params = {
#         'ln_Rdisk': ln_Rdisk[i],
#         'ln_sigmaLz': ln_sigmaLz[i],
#     }
#     start = time.perf_counter()
#     logP = logL_numpyro(data_generated, params, **aux_params)
    
#     logL_ls.append(jnp.sum(logP))
#     end = time.perf_counter()
#     # print('params', params)
#     print(f"time taken: {end - start:.4f} s")


# prior_distribution['lnL'] = jnp.array(logL_ls)
# file = '/data/hz420-2/radial_migration_kernel/mock_sample/L_conditioned/Prior_distribution_withlnL.pkl'
# with open(file, 'wb') as f:
#     pickle.dump(prior_distribution, f)

N_knots = 8
aux_knots = jnp.linspace(0.,15.,N_knots)
aux_params = {'R_scale_for_sampling':R_scale_for_sampling, 
            'F_scale_for_sampling':F_scale_for_sampling, 
            'F_centre_for_sampling':F_centre_for_sampling}  # Auxiliary parameters for the model
aux_params['aux_knots'] = jnp.linspace(0.,15.,N_knots)

params= {'ln_Rdisk': jnp.array([ 1.0166726 ,  1.0310085 ,  1.3933964 ,  0.7371091 , -0.90100026,
        0.2041145 ,  0.08324957,  1.1941382 ]), 'ln_sigmaLz': jnp.array([4.0198526, 2.902585 , 5.4031043, 5.091751 , 5.7446647, 6.606856 ,
       7.7502027, 5.8808165])}

# params = {'ln_Rdisk': jnp.array([ 1.0129285 ,  1.1170565 ,  1.1829929 ,  1.0032728 , -0.00879301,
#        -0.89356416,  0.7450388 ,  1.0250543 ]), 
#        'ln_sigmaLz': jnp.array([5.6591077, 3.8715577, 3.332939 , 4.028008 , 6.5052104, 6.417953 ,
#        6.1114144, 5.9913044])
#        }

# params = {'ln_Rdisk': jnp.array([ 1.0087975 ,  1.079981  ,  1.125587  ,  1.0048282 ,  0.3168702 ,
#        -0.28212965,  0.8284001 ,  1.0168505 ]), 'ln_sigmaLz': jnp.array([5.7696385, 4.553473 , 4.182361 , 4.624579 , 6.39234  , 6.589011 ,
#        6.118149 , 5.989861 ])}

# age_pivot = np.array([0, 4, 8, 12, 20])
# sigmaLz_pivot = np.array([10, 200, 400, 1200, 2000]) # sigma_Lz at 6 Gyr
# sigmaLz_gt_interp = sp.interpolate.interp1d(age_pivot, sigmaLz_pivot, kind='cubic', fill_value='extrapolate', bounds_error=False)
# Rd_gt = Rd_evolution_jump(np.linspace(0,12,100), Rdmax = 3.45, Rdmin = 1, tau_Rd = 7.0, delta_tau_Rd = 1.0)
# Rd_gt_interp = sp.interpolate.interp1d(np.linspace(0,12,100), Rd_gt, kind='cubic', fill_value='extrapolate', bounds_error=False)
# ln_Rdisk_knots = jnp.log(Rd_gt_interp(aux_params['aux_knots']))  # Convert Rd to ln(Rdisk)
# ln_sigmaLz_knots = jnp.log(sigmaLz_gt_interp(aux_params['aux_knots']))  # Convert sigma_Lz to ln(sigma_Lz)
# params = {
#     'ln_Rdisk': ln_Rdisk_knots,
#     'ln_sigmaLz': ln_sigmaLz_knots,
# }
#logL: 29552.21484375

for i in tqdm(range (0, 100)):
    start = time.perf_counter()
    logP = logL_numpyro(data_generated, params, **aux_params)
    end = time.perf_counter()
    # print('params', params)
    print(f"time taken: {end - start:.4f} s")
    # print('logL: ', jnp.sum(logP))
# print(data_generated)