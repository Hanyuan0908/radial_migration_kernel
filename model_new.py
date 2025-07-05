import jax.numpy as jnp
import jax.scipy as jsp
import jax
jax.config.update("jax_enable_x64", True)
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
import scipy as sp
import matplotlib as mpl
import cmasher as cmr
from astropy.io import fits
from astropy.table import Table

from tqdm import tqdm
import time
import agama
agama.setUnits(mass=1, length=1, velocity=1)

Vc0=240.

def scalarize(x):
    return x if len(x) > 1 else x[0]
def v_circ(pot, r):
    return scalarize(-r * pot.force(np.column_stack((r, r*0, r*0)))[:,0]) ** 0.5

def exponential_pdf_log(x, scale):
    """
    Probability density function of an exponential distribution.
    
    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF. Can be scalar or array.
    scale : float or array_like
        The scale parameter (1/λ) of the exponential distribution. Must be > 0.
    
    Returns
    -------
    pdf : array_like
        The PDF values at x.
    """
    # Ensure scale > 0
    scale = jnp.asarray(scale)
    # PDF formula: (1/scale) * exp(–x/scale) for x >= 0, else 0
    pdf = jnp.log(1.0 / scale) + (-x / scale)
    return jnp.where(x >= 0, pdf, -jnp.inf)

def XexpX_pdf_log(x, a):
    """
    Probability density function of the distribution proportional to x * exp(-x/a).
    
    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF. Can be scalar or array.
    a : float
        Scale parameter > 0.
    
    Returns
    -------
    pdf : array_like
        The PDF values at x.
    """
    # Ensure a > 0
    a = jnp.asarray(a)
    # PDF formula: (1/a^2) * x * exp(-x/a)
    pdf = jnp.log(x) - jnp.log(a**2) - (x / a)
    return jnp.where(x >= 0, pdf, -jnp.inf)

def sample_x_exp(xp, a, size):
    """
    Draw `size` samples from the PDF f(x) ∝ x * exp(-x/a), x>=0.
    
    Parameters
    ----------
    xp : numpy.random.Generator, e.g. np.random.default_rng()
    a  : float, scale parameter > 0
    size : int or tuple, number of samples
    
    Returns
    -------
    samples : array of shape `size`
    """
    # shape=k, scale=theta => Gamma(k, theta)
    return xp.gamma(shape=2.0, scale=a, size=size)


def lnG(x, mu, s):
    """
    Log of Gaussian distribution

    Args:
        x (float or array): Value(s) at which to evaluate the log of the Gaussian
        mu (float): Mean of the Gaussian
        s (float): Standard deviation of the Gaussian
    """
    return -.5*(x-mu)**2/s**2 - .5*jnp.log(2.*jnp.pi*s**2)

def Rd_evolution(age, Rdmax = 3.45, Rdmin = 2.31, tau_Rd = 9.0, delta_tau_Rd = 1.0):

    val = Rdmax - (Rdmax-Rdmin)/2 * (jnp.tanh((age-tau_Rd)/delta_tau_Rd)+1)

    return val

def getCylindricalFromCartesian_clockwise(x, y, vx, vy):

    R = np.sqrt(x**2+y**2)
    phi = np.arctan2(y, x)

    vR = vx*x/R + vy*y/R
    vphi = -vy*x/R + vx*y/R

    return R, phi, vR, vphi

''' Inside-out formation prescription '''


def Rd_evolution_jump(age, Rdmax = 3.45, Rdmin = 2.31, tau_Rd = 9.0, delta_tau_Rd = 1.0):

    val = Rdmax - (Rdmax-Rdmin)/2 * (jnp.tanh((age-tau_Rd)/delta_tau_Rd)+1)

    return val

def Rd_evolution_linear(age, tmax = 11.5, Rd_min = 1, gradient = 2.5):
    '''
    tmax: Maximum age of the disk
    Rd_min: Minimum disk scale length at t_lb = tmax
    gradient: gradient of the linear
    '''
    tbirth = tmax - age
    return Rd_min+(tbirth/tmax)*gradient


df = pd.read_csv(f"/data/hz420-2/bar_deceleration/Birth_radii_calculation.csv")
_age, MH_grad, MH_center = df.to_numpy().T
MH_grad_interp = jsp.interpolate.RegularGridInterpolator((_age,), MH_grad, fill_value = None, bounds_error=False)
MH_center_interp = jsp.interpolate.RegularGridInterpolator((_age,), MH_center, fill_value = None, bounds_error=False)

def Rb_calculator(age, MH):
    return (MH - MH_center_interp(age))/MH_grad_interp(age)

def MH_evolution_Lu24(age, Lbirth):

    Rbirth = Lbirth/Vc0
    feh = MH_center_interp(age) + MH_grad_interp(age)*Rbirth

    return jnp.where(jnp.isnan(feh), -jnp.inf, feh)

def MH_evolution_Lu24_brokenlaw(age, Lbirth, grad_inner = -0.05, R_broken = 5):

    Rbirth = Lbirth/Vc0

    feh1 = MH_center_interp(age) + grad_inner*Rbirth
    feh2 = MH_center_interp(age) + grad_inner*R_broken + MH_grad_interp(age)*(Rbirth-R_broken)

    feh = np.where(Rbirth<R_broken, feh1, feh2)

    return feh

def MH_evolution_sharma21(age, Lbirth, Fmin = -0.85, FR = -0.08, RF = 6.5, tauF = 3.2, taumax = 13):

    Rb = Lbirth/Vc0

    def Fmax(Rb):
        return Fmin*np.tanh(FR*(Rb-RF)/Fmin)
    
    val = Fmin + (Fmax(Rb)-Fmin) * np.tanh((taumax-age)/tauF)

    return val

def MH_evolution_Frankel20(age, Lbirth, MH_max = 0.7, gamma = 0.456, MH_grad = -0.0936, MH_grad_inner = -0.03):

    MH_max_time = MH_max * (1 - age/12)**gamma

    b_mh_inner = 0
    b_mh = (MH_grad_inner - MH_grad) * 3

    D_mh_inner = MH_grad_inner * Lbirth/Vc0
    D_mh = MH_grad * Lbirth/Vc0

    # print(MH_max_time + D_mh_inner + b_mh_inner - (MH_max_time + D_mh + b_mh))
    feh = jnp.where(Lbirth<3*Vc0, MH_max_time + D_mh_inner + b_mh_inner, MH_max_time + D_mh + b_mh)

    return jnp.where(np.isnan(feh), -jnp.inf, feh)




def sample_XexpX(xp, a, size):
    """
    Draw `size` samples from the PDF f(x) ∝ x * exp(-x/a), x>=0.
    
    Parameters
    ----------
    xp : numpy.random.Generator, e.g. np.random.default_rng()
    a  : float, scale parameter > 0
    size : int or tuple, number of samples
    
    Returns
    -------
    samples : array of shape `size`
    """
    # shape=k, scale=theta => Gamma(k, theta)
    return xp.gamma(shape=2.0, scale=a, size=size)





def fL0_log(L,Lcentre):
    '''
    L: birth AM
    Lcentre: scale length of the initial AM distribution
    '''
    return jnp.log(L)+(-L/Lcentre)

def kernel_SB15_log(L, Lp, Lcenter, sigmaLz):
    ''' 
    f(L|age, Lbirth) from Sanders & Binney 2015
    '''
    # sigmaLz = sigmaLz_func(age)
    sigmaLz = jnp.where(sigmaLz > 5, sigmaLz, 5)  # Avoid division by zero
    Lcenter = jnp.where(Lcenter > 5, Lcenter, 5)  # Avoid division by zero
    val1 = Lp - sigmaLz**2/(2*Lcenter)

    return lnG(L, val1, sigmaLz) #-0.5 * val1**2 / sigmaLz**2 - 0.5 * jnp.log(2*jnp.pi*sigmaLz**2)


def f_MH0_log(MH, age, Lbirth, tol = 5e-2):

    '''
    f(MH|age, Lbirth) = delta(MH - MH_evolution_XXX(age, Lbirth))
    '''

    # MH_evolution_Lu24(age, Lbirth), MH_evolution_Frankel20, MH_evolution_sharma21
    shape = age.shape
    age = age.reshape(-1)
    Lbirth = Lbirth.reshape(-1)
    MH_val = MH_evolution_Lu24(age, Lbirth)
    MH_val = MH_val.reshape(shape)

    return lnG(MH, MH_val, tol)




smoothing_scale = {'ln_Rdisk':0.4, 'ln_sigmaLz':1.2,}
@jax.jit
def smoothing_prior(params):
    lnP=0.
    if 'ln_Rdisk' in params:
        for i in smoothing_scale.keys():
            lnP += -jnp.sum(.5*(params[i][1:]-params[i][:-1])**2/(smoothing_scale[i])**2 + jnp.log(smoothing_scale[i]))

    return lnP

prior_scale_uniform = {'ln_Rdisk':[-2,2.3], 'ln_sigmaLz':[1., 8.],}
@jax.jit
def parameter_prior(params):
    '''
    Prior for the parameters
    '''
    lnP = 0.
    if 'ln_Rdisk' in params:
        lnP += jnp.sum(jnp.where((params['ln_Rdisk']>prior_scale_uniform['ln_Rdisk'][0]) & (params['ln_Rdisk']<prior_scale_uniform['ln_Rdisk'][1]), 0, -jnp.inf))
    if 'ln_sigmaLz' in params:
        lnP += jnp.sum(jnp.where((params['ln_sigmaLz']>prior_scale_uniform['ln_sigmaLz'][0]) & (params['ln_sigmaLz']<prior_scale_uniform['ln_sigmaLz'][1]), 0, -jnp.inf))

    return lnP
@jax.jit
def log_prior(params):
    '''
    Log prior for the parameters
    '''
    lnP = 0.
    lnP += parameter_prior(params)
    lnP += smoothing_prior(params)

    return lnP

# @jax.jit
# from functools import partial
# @partial(jax.jit, static_argnums=(2,3,4,5))
@jax.jit
def logL_numpyro(data, params, aux_knots, 
                 F_centre_for_sampling, 
                 F_scale_for_sampling, 
                 R_scale_for_sampling):

    time_start = time.perf_counter()

    L0_scale_for_sampling = R_scale_for_sampling * Vc0  # Scale for L0 sampling, assuming L0 is in units of Vc0


    F_sample_num = data['F_sample_num']
    F_sample_denom = data['F_sample_denom']
    L0_sample_num = data['L0_sample_num']
    L0_sample_denom = data['L0_sample_denom']
    age_sample_num = data['age_sample_num']
    age_sample_denom = data['age_sample_denom']
    L_sample_num = data['L_sample_num']
    L_sample_denom = data['L_sample_denom']
    weights = data['weights']

    N_sample = L_sample_denom.shape[0]
    N_star = L_sample_denom.shape[1]

    ln_sigmaLz_knots_knots = params['ln_sigmaLz']
    ln_Rdisk_knots_knots = params['ln_Rdisk']

    sigmaLz_knots = jnp.exp(ln_sigmaLz_knots_knots)
    Rdisk_knots = jnp.exp(ln_Rdisk_knots_knots)
    Lcentre_knots = Rdisk_knots * Vc0

    Lcentre_func =  InterpolatedUnivariateSpline(aux_knots, Lcentre_knots, k=3)
    sigmaLz_func = InterpolatedUnivariateSpline(aux_knots, sigmaLz_knots, k=3)

    #### numerator section ####
    Lcentre_sample = Lcentre_func(age_sample_num)
    sigmaLz_sample = sigmaLz_func(age_sample_num)

    logP_L_given_age_L0 = kernel_SB15_log(L_sample_num, L0_sample_num, Lcentre_sample, sigmaLz_sample)
    logP_F_given_age_L0 = f_MH0_log(F_sample_num, age_sample_num, L0_sample_num)
    logP_L0_given_age = fL0_log(L0_sample_num, Lcentre_sample)
    logP_L0_sample = XexpX_pdf_log(L0_sample_num, L0_scale_for_sampling)

    log_num = logP_L_given_age_L0 + logP_F_given_age_L0 + logP_L0_given_age - logP_L0_sample
    log_num_val = jsp.special.logsumexp(log_num, axis=0) - jnp.log(N_sample)

    #### denominator section ####

    Lcentre_sample = Lcentre_func(age_sample_denom)
    sigmaLz_sample = sigmaLz_func(age_sample_denom)

    logP_L_given_age_L0 = kernel_SB15_log(L_sample_denom, L0_sample_denom, Lcentre_sample, sigmaLz_sample)
    logP_F_given_age_L0 = f_MH0_log(F_sample_denom, age_sample_denom, L0_sample_denom)
    logP_L0_given_age = fL0_log(L0_sample_denom, Lcentre_sample)
    logP_L0_sample = XexpX_pdf_log(L0_sample_denom, L0_scale_for_sampling)
    logP_F_sample = lnG(F_sample_denom, F_centre_for_sampling, F_scale_for_sampling)

    log_denom = logP_L_given_age_L0 + logP_F_given_age_L0 + logP_L0_given_age - logP_L0_sample - logP_F_sample
    log_denom_val = jsp.special.logsumexp(log_denom, axis=0) - jnp.log(N_sample)

    logL = (log_num_val - log_denom_val) * (weights)

    time_end = time.perf_counter()

    # jax.debug.print("param = {y}, log_p = {x}, time = {z}", x=jnp.sum(logL), y=params, z = (time_end - time_start))

    return logL


def generate_sample_for_MC_integration(data, R_scale_for_sampling = 6, 
                                       F_centre_for_sampling = -0.3, 
                                       F_scale_for_sampling = 1, 
                                       N_sample = int(1e4)):

    Vc0 = 240.
    Z = data['MH']
    log_age = data['log_age']
    Lz = data['Lz']
    sigma_Z = data['sigma_MH']
    sigma_log_age = data['sigma_logage']
    sigma_Lz = data['sigma_Lz']

    N_star = len(Z)

    jax_random_key1 = jax.random.PRNGKey(42)
    jax_random_key2 = jax.random.PRNGKey(10086)
    jax_random_key3 = jax.random.PRNGKey(10010)
    jax_random_key4 = jax.random.PRNGKey(999)
    jax_random_key5 = jax.random.PRNGKey(2025)
    jax_random_key6 = jax.random.PRNGKey(124)
    jax_random_key7 = jax.random.PRNGKey(456)
    jax_random_key8 = jax.random.PRNGKey(789)


    F_stack = jnp.repeat(Z[None, :], N_sample, axis=0)
    sigmaF_stack = jnp.repeat(sigma_Z[None, :], N_sample, axis=0)
    L_stack = jnp.repeat(Lz[None, :], N_sample, axis=0)
    sigmaL_stack = jnp.repeat(sigma_Lz[None, :], N_sample, axis=0)
    logage_stack = jnp.repeat(log_age[None, :], N_sample, axis=0)
    sigmalogage_stack = jnp.repeat(sigma_log_age[None, :], N_sample, axis=0)

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

    sample_generated = {
        'L0_sample_num': L0_sample1,
        'L0_sample_denom': L0_sample2,
        'F_sample_num': F_sample1,
        'F_sample_denom': F_sample2,
        'age_sample_num': age_sample1,
        'age_sample_denom': age_sample2,
        'L_sample_num': L_sample1,
        'L_sample_denom': L_sample2,
        }
    return sample_generated


def generate_aux_knots(Nknots = 10, age_max = 12.):
    '''
    Generate auxiliary knots for the model
    '''
    aux_knots = jnp.linspace(0., age_max, Nknots-1)
    aux_knots = jnp.append(aux_knots, 15.)

    return aux_knots

#%%

# @jax.jit
# def logP_F_L_given_tau(F, L, age, param, aux_knots = None, N_sample = int(1e4)):

#     N_star = len(F)
#     jax_random_key = jax.random.PRNGKey(0)
#     N_sample = int(1e4)
#     R_scale_for_sampling = 6
#     ln_sigmaLz_knots_knots = param['ln_sigmaLz']
#     ln_Rdisk_knots_knots = param['ln_Rdisk']

#     sigmaLz_knots = jnp.exp(ln_sigmaLz_knots_knots)
#     Rdisk_knots = jnp.exp(ln_Rdisk_knots_knots)
#     Lcentre_knots = Rdisk_knots * Vc0

#     Lcentre_func =  InterpolatedUnivariateSpline(aux_knots, Lcentre_knots, k=3)
#     sigmaLz_func = InterpolatedUnivariateSpline(aux_knots, sigmaLz_knots, k=3)
    
#     L0_sample = jax.random.exponential(jax_random_key, shape=(N_sample,N_star)) * (R_scale_for_sampling * Vc0) # * Lcentre
#     # L0_sample = jax.random.uniform(jax_random_key, shape=(N_sample,N_star)) * 50*Vc0
    
#     F_stack = jnp.repeat(F[None, :], N_sample, axis=0)
#     L_stack = jnp.repeat(L[None, :], N_sample, axis=0)
#     age_stack = jnp.repeat(age[None, :], N_sample, axis=0)

#     Lcentre_sample = Lcentre_func(age_stack)
#     sigma_Lz_sample = sigmaLz_func(age_stack)

#     logP1 = kernel_SB15_log(L_stack, L0_sample, Lcentre_sample, sigma_Lz_sample)
#     logP2 = f_MH0_log(F_stack, age_stack, L0_sample)
#     logP3 = fL0_log(L0_sample, Lcentre_sample)
#     logP_L0 = exponential_pdf_log(L0_sample, (R_scale_for_sampling * Vc0))

#     log_val = logP1 + logP2 + logP3  - logP_L0 #+ jnp.log(50*Vc0)#
#     log_val_sum = jax.scipy.special.logsumexp(log_val, axis=0) - jnp.log(N_sample)  # Normalize by the number of samples

#     return log_val_sum

# @jax.jit
# def logP_L_given_tau(L, age, param, aux_knots = None, N_sample = int(1e4)):

#     N_star = len(age)
#     jax_random_key1 = jax.random.PRNGKey(42)
#     jax_random_key2 = jax.random.PRNGKey(116)
#     N_sample = int(1e4)
#     R_scale_for_sampling = 6
#     F_scale_for_sampling = 1

#     ln_sigmaLz_knots_knots = param['ln_sigmaLz']
#     ln_Rdisk_knots_knots = param['ln_Rdisk']

#     sigmaLz_knots = jnp.exp(ln_sigmaLz_knots_knots)
#     Rdisk_knots = jnp.exp(ln_Rdisk_knots_knots)
#     Lcentre_knots = Rdisk_knots * Vc0

#     Lcentre_func =  InterpolatedUnivariateSpline(aux_knots, Lcentre_knots, k=3)
#     sigmaLz_func = InterpolatedUnivariateSpline(aux_knots, sigmaLz_knots, k=3)

#     L0_sample = jax.random.exponential(jax_random_key1, shape=(N_sample,N_star)) * (R_scale_for_sampling * Vc0) # * Lcentre
#     F_sample = jax.random.normal(jax_random_key2, shape=(N_sample,N_star)) * F_scale_for_sampling + (-0.5)  # Assuming a standard normal distribution for F
#     # L0_sample = jax.random.uniform(jax_random_key1, shape=(N_sample,N_star)) * 50*Vc0  # Uniform between 10 and 11 * Vc0
#     # F_sample = jax.random.uniform(jax_random_key2, shape=(N_sample,N_star)) * 6 - 3  # Uniform between -0.5 and 2.5

#     L_stack = jnp.repeat(L[None, :], N_sample, axis=0)
#     age_stack = jnp.repeat(age[None, :], N_sample, axis=0)

#     Lcentre_sample = Lcentre_func(age_stack)
#     sigma_Lz_sample = sigmaLz_func(age_stack)
#     # jax.debug.print("Lcentre_sample = {x}", x=Lcentre_sample)

#     logP1 = kernel_SB15_log(L_stack, L0_sample, Lcentre_sample, sigma_Lz_sample)
#     logP2 = f_MH0_log(F_sample, age_stack, L0_sample)
#     logP3 = fL0_log(L0_sample, Lcentre_sample)
#     logP_L0 = exponential_pdf_log(L0_sample, (R_scale_for_sampling * Vc0))
#     logP_F = lnG(F_sample, -0.5, F_scale_for_sampling)  # Assuming a standard normal distribution for F
    
#     log_val = logP1 + logP2 + logP3 - logP_L0 - logP_F# + jnp.log(50*Vc0) + jnp.log(6)#
#     # jax.debug.print("log_val max = {x}", x=jnp.amax(log_val))
#     log_val_sum = jax.scipy.special.logsumexp(log_val, axis=0) - jnp.log(N_sample)  # Normalize by the number of samples

#     return log_val_sum

# @jax.jit
# def logP_F_given_tau_L(F, L, age, param, aux_knots = None, N_sample = int(1e4)):

#     logP_F_L_given_tau_val = logP_F_L_given_tau(F, L, age, param, aux_knots, N_sample)
#     logP_L_given_tau_val = logP_L_given_tau(L, age, param, aux_knots, N_sample)

#     logP_F_given_tau_L_val = logP_F_L_given_tau_val - logP_L_given_tau_val

#     return logP_F_given_tau_L_val