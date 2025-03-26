import numpy as np
import pandas as pd
import funcs
from scipy.interpolate import CubicSpline
import astropy.cosmology
from tqdm import tqdm
import matplotlib.pyplot as plt

def FractionalMasssLost(alpha, Mc, Delta):
    """
    Computes the fractional GC mass loss 
    
    K = rho_GC,0 / rho_GC
    
    """    
    # Compute GC formation rate integrals
    top = funcs.GCDensityInitialInt(alpha, Mc)[0]
    bottom = funcs.GCDensityInt(alpha, Delta, Mc)[0]
    return top/bottom

def phiMetallicity(tf0, Z, Zsolar=0.0134, disp=0.25):
    """Computes the cluster formation metallicity distribution"""
    # Define the mean metallicity at given redshift 
    Zmean = -0.017*(tf0/1e9)**2 + 0.04559*(tf0/1e9) + 0.42267

    phiZ = Z*np.exp(-(np.log10(Z/Zsolar)-Zmean)**2/(2*disp**2))
    return phiZ

def phiRedshift(redshift, file='data/formationEB.txt'):
    """Compute the phi_z term using the cubic spline"""
    # Load the data for the spline fitting
    xfit,yfit = np.loadtxt(file, unpack=True)
    
    # Interpolate with cubic spline
    spline = CubicSpline(xfit, yfit)
    phi_z = spline(redshift) # Evaluate at the given redshifts

    return phi_z.sum()

def Normalisation(file, alpha, Mc):
    """Computes the merger rate normalisation factor"""
    
    # Load the input file for this simulation to get the simulated cluster properties
    with open(file, 'r') as f:
        Mcl = f.readline().split(':')[1].strip('\n [ ]')
        Mcl = np.asarray(Mcl.split(', '), dtype=float)

        Metal = f.readline().split(':')[1].strip('\n [ ]')
        Metal = np.asarray(Metal.split(', '), dtype=float)
        
        FormTime = f.readline().split(':')[1].strip('\n [ ]')
        FormTime = np.asarray(FormTime.split(', '), dtype=float)

        redshift = f.readline().split(':')[1].strip('\n [ ]')
        redshift = np.asarray(redshift.split(', '), dtype=float)
    

    # Initial GC formation rate
    phiGC = funcs.GCDensityInitial(Mcl, alpha, Mc)*Mcl
    phiGCTot = phiGC.sum()

    ### Metallicity
    phiZTot = 0
    for tf0 in FormTime:
        phiZ = phiMetallicity(tf0, Metal)
        phiZTot+=phiZ.sum()

    ### Redshift
    phiRedTot = phiRedshift(redshift)

    return phiRedTot*phiZTot*phiGCTot

def ComputeMergerRate(file, data, alpha, Mc, Delta):
    '''Compute the merger rate at look back time tau'''
    # Load the input file for this simulation to get the simulated cluster properties
    with open(file, 'r') as f:
        Mcl = f.readline().split(':')[1].strip('\n [ ]')
        Mcl = np.asarray(Mcl.split(', '), dtype=float)

        Metal = f.readline().split(':')[1].strip('\n [ ]')
        Metal = np.asarray(Metal.split(', '), dtype=float)
        
        FormTime = f.readline().split(':')[1].strip('\n [ ]')
        FormTime = np.asarray(FormTime.split(', '), dtype=float)

        redshift = f.readline().split(':')[1].strip('\n [ ]')
        redshift = np.asarray(redshift.split(', '), dtype=float)

    # Group the data to find the merger rate at each configuration
    grouped = data.groupby(['tlbform','Z','Mcl']).size()

    total=0
    for idx in tqdm(grouped.index):
        tmpVal = grouped.loc[idx]*idx[2]

        # Find corresponding redshift
        tmp_redshift=redshift[FormTime==idx[0]/1e9]
        tmpVal*=phiRedshift(tmp_redshift)

        # find phicl
        tmpVal*=funcs.GCDensityInitial(idx[2], alpha, Mc)

        # find phi Z
        tmpVal*=phiMetallicity(idx[0],idx[1])
        total+=tmpVal

    print(total, FractionalMasssLost(alpha, Mc, Delta), funcs.GCDensityInt(alpha, Delta, Mc)[0])
    # Multiply by fractional mass loss
    total = total * FractionalMasssLost(alpha, Mc, Delta) * funcs.GCDensityInt(alpha, Delta, Mc)[0]

    # Compute Norm
    norm = Normalisation(file, alpha, Mc)

    return total/norm

def ForParam(param='chi_f', *args):
    '''Computes the local merger rate for a given parameter'''
    initFile, data, alpha, Mmax, Delta = args

    if param=='chi_f':
        # Test with chi_eff
        dchi=0.1
        chibins=np.arange(-1, 1,)
        heights= []
        for chi in range(chibins.size-1):
            tmpData = data.loc[(data.chi_f>=chibins[chi])&(data.chi_f<chibins[chi+1])]
            val = ComputeMergerRate(initFile, tmpData, alpha, Mmax, Delta)
            heights.append(val)
            
        heights = np.asarray(heights)/dchi

        return chibins, heights
    
    if param=='Mass':
        # Test with Mass
        dm=10
        massbins=np.arange(0, 100, dm)
        heights= []
        for mass in range(massbins.size-1):
            tmpData = data.loc[(data.m1>=massbins[mass])&(data.m1<massbins[mass+1])]
            val = ComputeMergerRate(initFile, tmpData, alpha, Mmax, Delta)
            heights.append(val)
        
        heights = np.asarray(heights)/dm

        return massbins, heights, dm
    

if __name__=='__main__':
    # Load the data from cBHBd
    data = pd.read_csv('Testing/mergers_1.txt',delimiter='\t')

    # Load the GC function values
    Delta, Mmax, alpha = np.loadtxt('data/last_chain_x1.txt', unpack=True)

    Mmax=10**Mmax
    Delta=10**Delta

    '''For testing I will just assume the first value'''    
    Delta=Delta[0]
    Mmax=Mmax[0]
    alpha=2

    # Keep only mergers happening in the local universe (out to detector volume)
    data = data.loc[(data.tlbform-data.mergertime>=0)&(data.tlbform-data.mergertime<=2e9)]

    ## Initial conditions file --> change as necessary
    initFile = 'Testing/ClusterInput_Seeds-1-1.txt'

    chibins, heights, dparam = ForParam('Mass', initFile, data, alpha, Mmax, Delta)

    plt.bar(chibins[:-1], heights, dparam)

    plt.show()




