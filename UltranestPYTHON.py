#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 21:25:28 2024

@author: marcusbedford
"""

import ultranest
import numpy as np
from astropy.table import Table
from scipy.interpolate import LinearNDInterpolator
import scipy


def UltranestMSTO(sobjectIDs):
    """
    takes an Sobject ID and samples the parametre space of log(age) and mass 1,600,000 times, 
    comparing theoretical stars to observed stars.

    Parameters:
    - sobject_id (int): The indentifying number for a star.

    Returns: None
    - Saves Ultranest's standard output to a folder labeled with the stars Sobject ID
    """
    galah_dr4_raw = Table.read('parsecandgalah/galah_dr4_allstar_240207.fits')
    parsec = Table.read('parsecandgalah/parsec_isochrones_extended_marcus.fits')
    '''
    uncomment this if this ends up being the method for cutting GALAH to just be MSTO
    ##MSTO stars
    MSTOstarsCOORDS = (
        (galah_dr4_raw["teff"] > 5100) & (galah_dr4_raw["teff"] < 5900) & 
        ((galah_dr4_raw["phot_g_mean_mag"] - 2.5*np.log10((galah_dr4_raw["r_med"]/10)**2))<4.3) &
        ((galah_dr4_raw["phot_g_mean_mag"] - 2.5*np.log10((galah_dr4_raw["r_med"]/10)**2))>2.4)
    )
    galahMSTO = galah_dr4_raw[MSTOstarsCOORDS]
    '''
    ##PARSEC cuts to speed up interpolator 
    teff_er = 70
    MSTOcuts = (
        (10**parsec["logT"] > (5100-2*teff_er)) & (10**parsec["logT"] < (5900+2*teff_er)) & 
        (parsec["logg"]<(4.69+0.2)) & (parsec["logg"]>(2.01+0.2))
    )
    cutParsec = parsec[MSTOcuts]
    
    ##interpolator set up. can take a couple of minutes. may be faster to pickle
    smallpoints = np.array([cutParsec['mass'],cutParsec['logAge'],cutParsec['m_h']]).T
    smallvalues = np.array([cutParsec['logT'],cutParsec['logg'],cutParsec['logL']]).T
    parsec_interpolator_small = LinearNDInterpolator(
        smallpoints,
        smallvalues 
    )


    ##function that finds the global minimum 
    def initialGuesser(sobject_id):
        starnum = (galah_dr4_raw["sobject_id"]==sobject_id)
        star = galah_dr4_raw[starnum][0]
        m_h = star["fe_h"]
        e_m_h = star["e_fe_h"]
        
        age, mass = star["age"], star["mass"]
        print(mass,age)
        x0 = np.array([mass,np.log10(age*10**9)])
        args = {"args": (star["teff"], star["logg"],np.log10(star["lbol"]), m_h, e_m_h)}
        results = scipy.optimize.basinhopping(log_post_lum2,x0,minimizer_kwargs=args,niter=200)
        return results.x

    def my_likelihood(params):
        mass, logage = params
        # compute intensity at every x position according to the model
        
        ##checks to see that our values are within a certain range of our PARSEC points to stop the interpolator funny buseiness.
        relevant_isochrone_points = (
            (np.abs(cutParsec['mass']-mass)<0.04) &
            (np.abs(cutParsec['logAge']-logage)<0.04) &
            (np.abs(cutParsec['m_h']-m_h)<0.5)   
        )
        if (True not in relevant_isochrone_points):
            return -10000000
        modelStar = parsec_interpolator_small(mass,logage,np.random.normal(m_h,e_m_h))
        model_values = np.array([10**modelStar[0],modelStar[1],modelStar[2]])
        
        # compare model and data with gaussian likelihood:
        like = -0.5 * (((model_values - input_values)/errors)**2).sum()
        if (np.isnan(like)):
            return -10000000
        return like

    def my_prior_transform(cube):
        params = cube.copy()
    
        # transform location parameter: uniform prior
        ##prior is updated to surround the initla guess
     
        lo = (inipos[0]-0.3)
        hi = (inipos[0]+0.3)
        params[0] = cube[0] * (hi - lo) + lo
        
        lo = (inipos[1]-0.35)
        hi = (inipos[1]+0.35)
        params[1] = cube[1] * (hi - lo) + lo
    
    
        return params

    param_names = ['mass', 'log(age)']
    starnum = (galah_dr4_raw["sobject_id"]==sobjectIDs)
    star = galah_dr4_raw[starnum][0]
    ####################################################
    ##below is the code for Running ultranest 
    inipos = initialGuesser(star["sobject_id"])
    m_h = star["fe_h"]
    e_m_h = star["e_fe_h"]
    input_values = np.array([star['teff'], star['logg'], np.log10(star['lbol'])])
    errors = np.array([star['e_teff'], star['e_logg'], 0.2])

    sampler = ultranest.ReactiveNestedSampler(param_names, my_likelihood, my_prior_transform, log_dir="MSTOStarResults/"+str(sobjectIDs), resume='overwrite')
    result = sampler.run(max_iters=5500,frac_remain=0.02,show_status=False)
    #sampler.print_results()
    
    ###################################################
    #below functions are for the intial guesser. essentially the same likelihood and prior used by the sampler but negative.
    def log_post_lum2(theta, teff, logg,lbol, m_h, e_m_h, e_teff=70, e_logg=0.1, e_loglum=0.2):
    
        prior = log_prior2(theta)
        if (np.isinf(prior) ):
            return prior
        like = log_like_lum2(theta, teff, logg,lbol, m_h, e_m_h, e_teff=70, e_logg=0.1, e_loglum=0.2)
        if np.isnan(like):
            return np.inf
      
        return like+prior
    def log_prior2(theta):
        if (theta[0] >= 0 and theta[0] <= 10 and theta[1]>=8 and theta[1]<=10.17):  # Prior range for a probability parameter
            return 0.0  # Uniform prior within the range
        else:
            return np.inf
    def log_like_lum2(theta, teff, logg, lbol,m_h, e_m_h, e_teff=70, e_logg=0.1, e_loglum=0.2):
        modelStar = parsec_interpolator_small(theta[0],theta[1],np.random.normal(m_h,e_m_h))
        
        #EXTRACT PARAMETRES FROM THE GIVEN PARSEC ROW
        model_values = np.array([10**modelStar[0],modelStar[1],modelStar[2]])
        input_values = np.array([teff, logg,lbol])
        
        errors = np.array([e_teff, e_logg,e_loglum])
        ##CALCULATE LIKELIHOOD
        
        differences = model_values - input_values 
        squared_differences = np.square(differences / errors)/2        
        ln_like = np.sum(squared_differences, axis=0)
        return ln_like