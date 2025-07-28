# radial_migration_kernel

Key scripts:

1. model_new.py contains all the source functions, including the data preprocessing, the loglikelihood function, and all its required ingredients.

2. numpyro_model.py warps the loglikelihood function to a model accessible by numpyro

3. run_optimisation_binned_MHmodel3.py runs the minimisation of the negative loglikelihood, and the behaviour is fine.

4. numpyro_optimisation.ipynb runs the NUTS sampler with numpyro. The behaviour is a little weird, as it doesn't seem to sample the degeneracy well.

5. test_numpyro_model.ipynb is the main results analysis and visualisation script, which could be ignored if one doesn't need it.

Key functions:

1. "logL_numpyro_withMHmodel3" is the best-performing loglikelihood of the age-metallicity plane fitting (fit P ([M/H] | age, Lz)) that takes the angular momentum diffusion (sigma_Lz), birth radial scale length (R_disk), and metallicity gradient (d[M/H] / dR) as functions of ages as the parameters.

2. "logL_numpyro4" is the same as above, but doesn't fit the metallicity gradient. It defaults to the metallicity model in Lu et al. (2024).

3. "generate_sample_for_MC_integration_withprob_samenumdenom" generate the samples for Monte Carlo integration for the loglikelihood given the raw measurement/mock data. The output of this function could be fed into the loglikelihood function directly.

Mock data:
1. mock_sample contains the mock dataset that will provide the necessary data to run the code, all one need to change in the files are the path to the mock_sample folder.
   
   1.1 mock_sample/L_conditioned/mock_data3.csv is the main mock sample I'm working with. The ground truth of angular momentum diffusion, initial disk scale length, and the metallicity gradient can be generated with the following code:

    age_pivot = np.array([0, 4, 8, 12, 20])
   
    sigmaLz_pivot = np.array([0, 300, 600, 1500, 2000])
   
    sigmaLz_gt_interp = sp.interpolate.interp1d(age_pivot, sigmaLz_pivot, kind='cubic', fill_value='extrapolate', bounds_error=False)
   
    Rd_gt = Rd_evolution_jump(np.linspace(0,15,100), Rdmax = 2.72, Rdmin = 1, tau_Rd = 6.0, delta_tau_Rd = 3.0)
   
    Rd_gt_interp = sp.interpolate.interp1d(np.linspace(0,15,100), Rd_gt, kind='cubic', fill_value='extrapolate', bounds_error=False)
   
    MH_grad_gt = (MH_evolution_Lu24(np.linspace(0,20,100), 9 * Vc0) - MH_evolution_Lu24(np.linspace(0,20,100), 8 * Vc0))
   
    ln_MH_grad_gt = np.log(-MH_grad_gt)
   
    ln_MH_grad_gt_interp = sp.interpolate.interp1d(np.linspace(0,20,100), ln_MH_grad_gt, kind='cubic', fill_value='extrapolate', bounds_error=False)


