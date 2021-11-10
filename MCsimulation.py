"""
Editor: Nina Chen
final version
11/10/2021
"""
#%% Import required module 
from skimage import io
from scipy import stats    
import numpy as np         
import pandas as pd        
import time                
import os                 

#%% Load the IRF information 
workpath  = os.getcwd() 
irfFileName = '/irf.tif'
irf = io.imread(workpath + irfFileName) 
avgIRF = np.sum(np.sum(irf, axis=1), axis=1)
avgIRF = avgIRF / np.max(avgIRF)

#%% Define the function for histogram simulation
def tDecaySimulation(IRF, I, A, tau1, tau2, photonCounts, res=256):
    '''
    This function is to simulate a single time decay histogram.
    Input:
        IRF         : Instrument Response Function
        I           : Intensity (Hz)
        A           : Fraction of longer lifetime
        tau1        : long lifetime
        tau2        : short lifetime
        photonCounts: Photon counts in this simulated decay histogram
        res         : Number of time bins (256 by default)
    Output:
        exponentialDecay    : Exponential decay curve before convolution
        simGroundTruthDecay : Ground truth decay curve after convolution
        simulatedDecay      : Simulated decay histogram after MC simulation
    '''
    # Obtain the time decay curve
    t = np.linspace(0, 50, res)                  # The time span from 0 ~ 50 ns
    component1 = A*np.exp(-t/tau1)               # Decay function for component 1
    component2 = (1-A)*np.exp(-t/tau2)           # Decay function for component 2
    exponentialDecay = I*(component1+component2) # Add two decay functions together
    simGroundTruthDecay = np.convolve(IRF, exponentialDecay) 
    simGroundTruthDecay = simGroundTruthDecay[:res]
    exponentialDecay_norm = exponentialDecay / max(exponentialDecay)  
    simGroundTruthDecay_norm = simGroundTruthDecay / max(simGroundTruthDecay) 
    area = sum(simGroundTruthDecay)              # Calculate the area under the function
    simGroundTruthDecay_pdfnorm = simGroundTruthDecay / area  # Normalize the decay function to have area under curve as 1 
    
    # Perform Monte Carlo method to draw the samples
    xk = np.arange(res)                      # xk represents time bins
    pk = simGroundTruthDecay_pdfnorm         # pk represents corresponding decay histogram (PDF)
    pk[-1] = 1-sum(pk[:-1])                  # Make sure the sum of pk equals to 1
    # Extract the samples from xk based on pk as pdf, size represents number of samplesto be extracted
    samples = stats.rv_discrete(name='samples', values=(xk, pk)).rvs(size=photonCounts)
    # Convert the extracted samples into histogram
    simulatedDecay = np.histogram(samples, bins=np.arange(-0.5, res, 1))[0] 
    simulatedDecay_norm = simulatedDecay / max(simulatedDecay)
    return exponentialDecay_norm, simGroundTruthDecay_norm, simulatedDecay_norm

def tDFLIMSimulation(irf, A_choice, tau1_choice, tau2_choice, pc_choice = [50, 100, 500, 1500, 5000],
                     res=256, n_duplicates=100):
    '''
    This function is to simulate a FLIM images raw data with the dimension (ex: 
        512 (pixels) x 512 (pixels) x 256 (bins))).
    Input:
        irf            : Instrument Response Function
        A_choice       : Fraction of longer lifetime
        tau1_choice    : long lifetime
        tau2_choice    : short lifetime
        res            : Number of time bins (256 by default)
    Output:
        IRF            : Instrument Response Function
        FLIMSimulation : Simulated FLIM images with histogram for each pixel 
        FLIM           : Simulated FLIM images (Average lifetime)
        FLIMTau1       : Simulated FLIM images (Longer lifetime)
        FLIMTau2       : Simulated FLIM images (Shorter lifetime)
        FLIMA          : Simulated FLIM images (Fraction of longer lifetime)
    '''
    print(60*'-')
    print('\nStart simulating the FLIM images ...')
    # Store training stats
    tic = time.clock() # Calculate time spent (Start)
    n_decays = len(A_choice) * len(tau1_choice) * len(tau2_choice) * len(pc_choice) * n_duplicates 
    
    FLIMSimulation = np.zeros((n_decays, res)) # Create an initialized array
    IRF = np.zeros((n_decays, res))
    FLIM = np.zeros((n_decays,))
    FLIMTau1 = np.zeros((n_decays,))
    FLIMTau2 = np.zeros((n_decays,))
    FLIMA = np.zeros((n_decays,))
    FLIMPC = np.zeros((n_decays,))
    I    = 1
    it   = 0
    for A in A_choice:
        for tau1 in tau1_choice:
            for tau2 in tau2_choice:
                for pc in pc_choice:
                    for dup in range(n_duplicates):
                        IRF[it, :] = avgIRF   # Store the information
                        expDecay, simGTDecay, simulatedDecay = tDecaySimulation(avgIRF, I, A, tau1, tau2, pc)  
                        FLIMSimulation[it, :] = simulatedDecay
                        FLIM[it] = A*tau1 + (1-A)*tau2
                        FLIMTau1[it] = tau1
                        FLIMTau2[it] = tau2
                        FLIMA[it] = A 
                        FLIMPC[it] = pc                          
                        it += 1
                    
    toc = time.clock() # Calculate time spent (End)
    elapseTime = toc - tic  # Calculate time spent
    print('\nSimulaton finished !!!')
    print('\nTime passed for simulating the images: {} seconds... '.format(elapseTime))
    print(60*'-')
    return IRF, FLIMSimulation, FLIM, FLIMTau1, FLIMTau2, FLIMA, FLIMPC

#%% Store the data in DataFrame
dup = 100
IRF, FLIMSimulation, FLIM, FLIMTau1, FLIMTau2, FLIMA, FLIMPC = tDFLIMSimulation(irf, [0.98, 0.99, 1.00], 
                                                                                list(np.linspace(0.9, 5.0, 42)),
                                                                                [0.5])  # Generate the simulated FLIM decay histogram
dataset = pd.DataFrame(columns=['IRF', 'TimeDecayHistogram', 'FLIM_A', # Create the dataframe to store the results
                            'FLIM_tau1', 'FLIM_tau2', 'photonCounts']) 
dataset = dataset.append({'IRF': IRF, 'TimeDecayHistogram': FLIMSimulation,  # Store the results
                          'FLIM_A': FLIMA, 'FLIM_tau1': FLIMTau1, 'FLIM_tau2': FLIMTau2, 
                          'photonCounts': FLIMPC}, ignore_index=True)
dataset.to_pickle(workpath + '/Simulation_test.pkl')  # Save the data
 