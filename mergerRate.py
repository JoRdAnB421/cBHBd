import numpy as np
import pandas as pd
import funcs
from scipy.interpolate import CubicSpline

def FractionalMasssLost(alpha, Mc, Delta):
    """
    Computes the fractional GC mass loss 
    
    K = rho_GC,0 / rho_GC
    
    """
    Delta=10**Delta
    Mc=10**Mc
    
    # Compute GC formation rate integrals
    top = funcs.GCDensityInitialInt(alpha, Mc)
    bottom = funcs.GCDensityInt(alpha, Delta, Mc)

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
    Mc = 10**Mc
    
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

    return phiRedTot,phiZTot,phiGCTot

if __name__=='__main__':
    # Load the data from cBHBd
    # data = pd.read_csv('Testing/mergers_1.txt',delimiter='\t')

    # Load the GC function values
    Delta, Mmax, alpha = np.loadtxt('data/last_chain_x1.txt', unpack=True)

    '''For testing I will just assume the first value'''    
    Delta=Delta[0]
    Mmax=Mmax[0]
    alpha=2

    # Keep only mergers happening in the local universe
    # data = data.loc[(data.tlbform-data.mergertime>=0)&(data.tlbform-data.mergertime<=2e9)]

    ## Testing compute of normalisation
    initFile = 'Testing/ClusterInput_Seeds-1-1.txt'
    val = Normalisation(initFile, alpha, Mmax)
    print(np.prod(val))


