# radial_migration_kernel

1. model_new.py contains the logL function "logL_numpyro" and all ingradient it need. All that logL_numpyro does is a series of matrix multiplication and summation

2. numpyro_model.py warps the loglikelihood function to a model accessible by numpyro

3. run_optimisation_binned.py & run_HMC_binned.py run the optimisation using minimiser and NUTS sampler respectively

4. evaluate_lnL_given_prior_binned.py exams the timing of the logL evaluation in a for loop, which does super weried things

5. test_logL.py examing the performance of the fitting

6. mock_sample contains the mock dataset that will provide the necessary data to run the code, all one need to change in the files are the path to the mock_sample folder


