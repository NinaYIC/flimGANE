"""
Editor: Nina Chen
Final version
Main code for flimGANE training and testing for FLIM
11/10/2021

"""

#%% Import required modules
import os
import numpy as np                    # Fundamental package for scientific compouting
import pandas as pd                   # For data manipulation and analysis
import random                         # Used to generate random number
import time
from skimage import io
import tensorflow as tf               # For DNN model development (backend)
import matplotlib.pyplot as plt       # To plot the figure    
plt.rcParams.update({'font.size': 30, 'axes.linewidth': 5, 'axes.titlepad': 25,
                     'axes.labelsize': 'medium',  'axes.labelpad': 25,
                     'figure.dpi': 100, 'savefig.dpi': 100, 
                     'xtick.major.size': 15, 'xtick.major.width': 5,
                     'xtick.major.pad': 20, 'xtick.direction': 'in',
                     'ytick.major.size': 15, 'ytick.major.width': 5,
                     'ytick.major.pad': 20, 'ytick.direction': 'in'})   
from keras import backend as K        # For DNN model development (Keras)
from keras.models import Model, load_model  # For model development
from keras.layers import Input, Dense, Lambda, Concatenate, Add
from keras.layers import Conv1D, Reshape, AveragePooling1D, Flatten
from keras.optimizers import Adam, RMSprop

workdir = os.getcwd()
res = 256
version = 'release_1'
savepath = workdir + "/Results/"

#%% Define the functions to be utilized
def generate_decay_histogram(IRF, A, tau1, tau2):
    '''
    This function is to simulate a single time decay histogram.
    Input:
        IRF         : Instrument Response Function
        A           : Fraction of longer lifetime
        tau1        : long lifetime
        tau2        : short lifetime
    Output:
        simGroundTruthDecay : Ground truth decay curve after convolution
    '''
    t = np.linspace(0, 50, res)
    component1 = A*np.exp(-t/tau1)
    component2 = (1-A)*np.exp(-t/tau2)
    exponentialDecay = component1+component2
    simGroundTruthDecay = np.convolve(IRF, exponentialDecay)
    simGroundTruthDecay = simGroundTruthDecay / max(simGroundTruthDecay)   
    return simGroundTruthDecay[:res]

def generate_data(Xtrain, IRF, FLIMA, FLIMTau1, FLIMTau2, n_samples=100000, display=False):
    '''
    This function is to generate the dataset for GAN model training
    Input:
        Xtrain      : Simulated decay histogram
        IRF         : Instrument Response Function
        FLIMA       : Fraction of longer lifetime
        FLIMtau1    : long lifetime
        FLIMtau2    : short lifetime
    Output:
        real        : Smooth decay curve (clear inputs)
        fake        : Simulated decay histogram (messy inputs)
        irf         : Instrument Response Function
        A           : Fraction of longer lifetime
        tau1        : long lifetime
        tau2        : short lifetime
    '''
    idx = random.sample(list(range(Xtrain.shape[0])),  n_samples)  # Obtain random indexes
    A = FLIMA[idx]
    tau1 = FLIMTau1[idx]
    tau2 = FLIMTau2[idx]
    irf, real, fake = np.zeros([n_samples, 256]), np.zeros([n_samples, 256]), np.zeros([n_samples, 256])
    fake = Xtrain[idx, :]
    irf = IRF[idx, :]
    for ind in range(n_samples):
        real[ind, :] = generate_decay_histogram(irf[ind, :], A[ind], tau1[ind], tau2[ind])   # Generate the smooth curve
        if ind % (n_samples/10) == 0 and ind != 0 and display:
            print(str(ind) + ' Samples Done!')    # Show the current status
    if display:
        print(str(n_samples) + ' Samples Done!')    
    return real, fake, irf, A, tau1, tau2

def generate_training_data(Xtrain, IRF, FLIMA, FLIMTau1, FLIMTau2, n_samples=100000):
    realinput, fakeinput, irf, A, tau1, tau2 = generate_data(Xtrain, IRF, FLIMA, FLIMTau1, 
                                                   FLIMTau2, n_samples=n_samples, display=False)
    valid = -np.ones((realinput.shape[0], 1))
    fake  = np.ones((realinput.shape[0], 1))
    return realinput, fakeinput, valid, fake, irf, A, tau1, tau2

def generate_flimGANEtraining_data(Xtrain, IRF, FLIMA, FLIMTau1, FLIMTau2, n_samples=100000):
    realinput, fakeinput, irf, A, tau1, tau2 = generate_data(Xtrain, IRF, FLIMA, FLIMTau1, 
                                                   FLIMTau2, n_samples=n_samples, display=False)
    return fakeinput, irf, A, tau1, tau2

def divide(tensors):
    output = tensors
    outputmax = K.max(output, axis=1, keepdims=True)
    import tensorflow as tf
    finalout = tf.math.divide(output, outputmax)
    return finalout

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred) 

def get_generative(G_in_decay, G_in_irf):
    G_in_decay_reshape = Reshape((256, 1))(G_in_decay)
    G_in_irf_reshape   = Reshape((256, 1))(G_in_irf)
    decayImage = Concatenate(axis=2)([G_in_decay_reshape, G_in_irf_reshape])
    decayImageConv1 = Conv1D(16, 4, padding='same', activation='relu')(decayImage)
    decayImagePool1 = AveragePooling1D(pool_size=2)(decayImageConv1)
    decayImageConv2 = Conv1D(16, 4, padding='same', activation='relu')(decayImagePool1)
    decayImagePool2 = AveragePooling1D(pool_size=2)(decayImageConv2)
    denseInput = Flatten()(decayImagePool2)
    attemptA    = Dense(16, activation='sigmoid')(denseInput) 
    G_out_A     = Dense(2, activation='relu', name="A_star")(attemptA)
    attemptTau1 = Dense(16, activation='sigmoid')(denseInput)
    G_out_tau1  = Dense(1, activation='relu', name="Tau1_star")(attemptTau1)
    attemptTau2 = Dense(16, activation='sigmoid')(denseInput)
    G_out_tau2  = Dense(1, activation='relu', name="Tau2_star")(attemptTau2)   
    G_out = Concatenate()([G_out_A, G_out_tau1, G_out_tau2])
    G_out = Dense(64, activation='sigmoid')(G_out)
    G_out = Dense(256, activation='tanh')(G_out)
    G_out = Add()([G_in_decay, G_out])
    G_out = Lambda(divide, output_shape=(256,))(G_out)
    G = Model(inputs=[G_in_decay, G_in_irf], outputs=G_out)
    G.compile(loss='mse', optimizer='adam')
    return G, G_out

def get_discriminative(D_in_decay):
    decay = Dense(128, activation='sigmoid')(D_in_decay)
    D_out =  Dense(64, activation='sigmoid')(decay)
    D_out =  Dense(8, activation='sigmoid')(D_out)
    D_out =  Dense(1)(D_out)
    D = Model(inputs=D_in_decay, outputs=D_out)
    D.compile(loss=wasserstein_loss, 
              optimizer=RMSprop(lr=0.00005),
              metrics=['accuracy'])
    return D, D_out

def get_estimative(E_in_decay, E_in_irf):
    decay = Dense(128, activation='sigmoid')(E_in_decay)
    irf   = Dense(64, activation='sigmoid')(E_in_irf)
    denseInput = Concatenate()([decay, irf])
    E_out = []
    attemptA    = Dense(32, activation='sigmoid')(denseInput)
    E_out_A     = Dense(1, activation='linear', name="D_A")(attemptA)
    attemptTau1 = Dense(32, activation='sigmoid')(denseInput)
    E_out_tau1  = Dense(1, activation='linear', name="D_Tau1_star")(attemptTau1)
    attemptTau2 = Dense(16, activation='sigmoid')(denseInput)
    E_out_tau2  = Dense(1, activation='linear', name="D_Tau2_star")(attemptTau2)
    E_out.append(E_out_A)  
    E_out.append(E_out_tau1)
    E_out.append(E_out_tau2)
    E = Model(inputs=[E_in_decay, E_in_irf], outputs=E_out)
    E.compile(loss=['mse','mse','mse'], optimizer='adam')
    return E, E_out

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def make_gan(GAN_in_decay, GAN_in_irf, G, D):
    set_trainability(D, False)
    x = G([GAN_in_decay, GAN_in_irf])
    GAN_out = D(x)
    GAN = Model([GAN_in_decay, GAN_in_irf], GAN_out)
    GAN.compile(loss=wasserstein_loss, 
                optimizer=RMSprop(lr=0.00005),
                metrics=['accuracy'])
    generator_model = Model([GAN_in_decay, GAN_in_irf], 
                            [GAN_out, x])
    generator_model.compile(loss=[wasserstein_loss, 'mse'],
                optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                loss_weights=[1, 500])
    return GAN, GAN_out, generator_model

def make_flimGANE(GAN_in_decay, GAN_in_irf, G, D):
    set_trainability(D, True)
    set_trainability(G, False)
    x = G([GAN_in_decay, GAN_in_irf])
    GAN_out = D([x, GAN_in_irf])
    GAN = Model([GAN_in_decay, GAN_in_irf], GAN_out)
    GAN.compile(loss='mse', optimizer='adam')
    return GAN, GAN_out

def sample_images(realinput, fakeinput, irf, epoch, G, r, c):

    x = fakeinput
    x_img = G.predict([x, irf])
    y = realinput
    t = np.linspace(0, 50, 256)
    
    fig, axs = plt.subplots(r, c, figsize=(75,30))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].plot(t, y[cnt, :], 'r', lw=7.5, label='High-count decay')
            axs[i,j].plot(t, x_img[cnt, :], 'bo--', markersize=15, label='Low-count decay')
            axs[i,j].legend()
            axs[i,j].set_ylim([-0.6, 1.2])
            cnt += 1
    
    saveFolder = workdir + "/Results/ver_" + version 
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    fig.savefig(saveFolder + "/decayHistogram_%d.png" % epoch)
    plt.close()

#%% Main code start here

#%% Start with generative model training
# Load the data set (This data is simulated based on IRF experiments for each pixel)
sim_filename = '/Simulation_version2_hela_090_500_100dups_test.pkl'
dataset = pd.read_pickle(workdir + sim_filename)
res = 256
Xtrain = np.reshape(dataset['TimeDecayHistogram'][0], (-1, res))
IRF = np.reshape(dataset['IRF'][0], (-1, res))
FLIMA = np.reshape(dataset['FLIM_A'][0], (-1, 1))
FLIMTau1 = np.reshape(dataset['FLIM_tau1'][0], (-1, 1))
FLIMTau2 = np.reshape(dataset['FLIM_tau2'][0], (-1, 1))
for i in range(np.shape(IRF)[0]):
    IRF[i, :] = IRF[i, :]/np.max(IRF[i, :])

# Create the generator model
G_in_decay = Input(shape=(res, ))
G_in_irf = Input(shape=(res, ))  
G, G_out = get_generative(G_in_decay, G_in_irf)
G.summary()

# Create the discriminative model
D_in_decay = Input(shape=(res, ))
D, D_out = get_discriminative(D_in_decay)
D.summary()

# Chained model (Combine generator and discriminator together)
GAN_in_decay = Input(shape=(res, ))
GAN_in_irf = Input(shape=(res, ))  
GAN, GAN_out, generator_model = make_gan(GAN_in_decay, GAN_in_irf, G, D)
generator_model.summary()

# Assign training parameters
verbose = True
v_freq = 50
epochs = 2000
d_loss = []
g_loss = []
generator_loss = []
n_critic = 5
clip_value = 0.01
e_range= range(epochs)

r, c = 3, 3
n_images = r * c
realinput_demo, fakeinput_demo, valid_demo, fake_demo, irf_demo, A_demo, tau1_demo, tau2_demo = generate_training_data(Xtrain, IRF, FLIMA, FLIMTau1, FLIMTau2, n_images)
# Create the new folder if not existed
saveFolder = workdir + "/Results/ver_" + version 
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)
# Save what are the selected simulated dataset from demo
np.save(saveFolder + "/realinput_demo.npy", realinput_demo) 
np.save(saveFolder + '/fakeinput_demo.npy', fakeinput_demo)
np.save(saveFolder + '/irf_demo.npy', irf_demo) 
np.save(saveFolder + '/A_demo.npy', A_demo) 
np.save(saveFolder + '/tau1_demo.npy', tau1_demo)
np.save(saveFolder + '/tau2_demo.npy', tau2_demo) 

# Train the model
for epoch in e_range:
    
    print("Epoch #" + str(epoch+1) + " .......")
    
    for _ in range(n_critic):

        # -------------------------
        #   Train Discriminator
        # -------------------------
    
        realinput, fakeinput, valid, fake, irf, A, tau1, tau2 = generate_training_data(Xtrain, IRF, FLIMA, FLIMTau1, FLIMTau2, 14000)
        
        imgs = realinput
        
        gen_imgs = G.predict([fakeinput, irf])
        
        d_loss_real = D.train_on_batch(imgs, valid)
        d_loss_fake = D.train_on_batch(gen_imgs, fake)
        d_loss_ = 0.5 * np.add(d_loss_real, d_loss_fake)
        for l in D.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            l.set_weights(weights)
        d_loss.append(d_loss_)
    
    # -------------------------
    #   Train Generator
    # -------------------------
    
    g_loss.append(generator_model.train_on_batch([fakeinput, irf], [valid, imgs]))
    
    generator_loss.append(G.test_on_batch([fakeinput, irf], imgs))
        
    if verbose and (epoch+1) % v_freq == 0:  # Show current progress for certain epochs
        print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch+1, g_loss[-1], d_loss[-1]))
        print("Epoch #{}: Generator's Loss: {}".format(epoch+1, generator_loss[-1]))

    if epoch % v_freq == 0:
        # Plot the demo and save the figure every v_freq epochs
        sample_images(realinput_demo, fakeinput_demo, irf_demo, epoch, G, r, c)
        
    if (epoch+1) % 100 == 0:
        # Save the model and results every 100 epochs
        G_name = 'WGAN_G_model_ver' + version + '_epoch' + str(epoch+1) + '.h5' 
        D_name = 'WGAN_D_model_ver' + version + '_epoch' + str(epoch+1) + '.h5' 
        GAN_name = 'WGAN_model_ver' + version + '_epoch' + str(epoch+1) + '.h5' 
        G.save(savepath + G_name) 
        D.save(savepath + D_name) 
        GAN.save(savepath + GAN_name)
        np.save(savepath + 'g_loss_ver' + version + '_epoch' + str(epoch+1) + '.npy', g_loss) 
        np.save(savepath + 'd_loss_ver' + version + '_epoch' + str(epoch+1) + '.npy', d_loss)
        np.save(savepath + 'generator_loss_ver' + version + '_epoch' + str(epoch+1) + '.npy', generator_loss)

# Save the model and results
G_name = 'WGAN_G_model_ver' + version + '.h5' 
D_name = 'WGAN_D_model_ver' + version + '.h5' 
GAN_name = 'WGAN_model_ver' + version + '.h5' 
G.save(savepath + G_name) 
D.save(savepath + D_name) 
GAN.save(savepath + GAN_name)
np.save(savepath + 'g_loss_ver' + version + '.npy', g_loss) 
np.save(savepath + 'd_loss_ver' + version + '.npy', d_loss)
np.save(savepath + 'generator_loss_ver' + version + '.npy', generator_loss)

#%% Start with estimative model training
# Load the IRF --> avg and normalize
irfFileName = '/irf.tif'
irf = io.imread(workdir + irfFileName)
avgIRF = np.sum(np.sum(irf, axis=1), axis=1)
avgIRF = avgIRF / np.max(avgIRF)
# Use the same simulation parameters as you did for the previous generative model training
tau1s = np.linspace(0.9, 5.0, 42)
tau2s = [0.5]
alphas = [0.98, 0.99, 1.00]
n_decays = len(tau1s) * len(tau2s) * len(alphas)
histograms = np.zeros((n_decays, res))
IRF = np.zeros((n_decays, res))
FLIMA = np.zeros((n_decays,))
FLIMtau1 = np.zeros((n_decays,))
FLIMtau2 = np.zeros((n_decays,))
it = 0
for tau1 in tau1s:
    for tau2 in tau2s:
        for alpha in alphas:
            histograms[it, :] = generate_decay_histogram(avgIRF, alpha, tau1, tau2)
            IRF[it, :] = avgIRF
            FLIMA[it] = alpha
            FLIMtau1[it] = tau1
            FLIMtau2[it] = tau2            
            it += 1
            
# Create the discriminative model
E_in_decay = Input(shape=(res, ))
E_in_irf = Input(shape=(res, ))  
E, E_out = get_estimative(E_in_decay, E_in_irf)
E.summary()

# Train the model
e_loss = []
epochs = 5000
verbose = True
v_freq = 10

tic = time.clock()
for epoch in range(epochs):
    print("Epoch #{}.......".format(epoch+1))    
    e_loss.append(E.train_on_batch([histograms, IRF], [FLIMA, FLIMtau1, FLIMtau2]))
    if verbose and (epoch+1) % v_freq == 0:  # Show current progress for certain epochs
        print("Epoch #{}: Discriminative Loss: {}".format(epoch+1, e_loss[-1]))
toc = time.clock()
elapseTime = toc = tic

# Assign the version to be saved and save the results
savepath = workdir + "/Results/"
E_name = 'E_model_ver' + version + '.h5' 
E.save(savepath + E_name) 
np.save(savepath + 'e_loss_ver' + version + '.npy', e_loss)

#%% Start flimGANE combinative training
# First, load the dataset if you haven't
sim_filename = '/Simulation_version2_hela_090_500_100dups_test.pkl'
dataset = pd.read_pickle(workdir + sim_filename)
res = 256
Xtrain = np.reshape(dataset['TimeDecayHistogram'][0], (-1, res))
IRF = np.reshape(dataset['IRF'][0], (-1, res))
FLIMA = np.reshape(dataset['FLIM_A'][0], (-1, 1))
FLIMTau1 = np.reshape(dataset['FLIM_tau1'][0], (-1, 1))
FLIMTau2 = np.reshape(dataset['FLIM_tau2'][0], (-1, 1))
for i in range(np.shape(IRF)[0]):
    IRF[i, :] = IRF[i, :]/np.max(IRF[i, :])

# Load the model
G = load_model(workdir + '/Example_generator.h5', custom_objects=dict(wasserstein_loss=wasserstein_loss))
E = load_model(workdir + '/Example_estimator.h5')
E.name = 'model_2_new'

# Chained model (Combine generator and discriminator together)
GAN_in_decay = Input(shape=(res, ))
GAN_in_irf = Input(shape=(res, ))  
flimGANE, flimGANE_out = make_flimGANE(GAN_in_decay, GAN_in_irf, G, E)
flimGANE.summary()

# Train the model
epochs = 500
n_crit = 100 
gan_loss = []
gan_val  = []
e_range= range(epochs)
# Separate dataset into training and validation (with ratio of 9:1)
valid_ind = random.sample(list(range(Xtrain.shape[0])),  int(Xtrain.shape[0]*0.1))
train_ind = [i for i in range(Xtrain.shape[0]) if i not in valid_ind]
A_valid = FLIMA[valid_ind]
tau1_valid = FLIMTau1[valid_ind]
tau2_valid = FLIMTau2[valid_ind]
irf_valid  = IRF[valid_ind, :]
X_valid    = Xtrain[valid_ind, :]
A_train = FLIMA[train_ind]
tau1_train = FLIMTau1[train_ind]
tau2_train = FLIMTau2[train_ind]
irf_train  = IRF[train_ind, :]
X_train_train    = Xtrain[train_ind, :]   
 
tic = time.clock()
for epoch in e_range:
    
    print("Epoch #" + str(epoch+1) + " .......")
    fakeinput, irf, A, tau1, tau2 = generate_flimGANEtraining_data(X_train_train, irf_train, 
                                                           A_train, tau1_train, tau2_train,
                                                           int(X_train_train.shape[0]*0.25))

    for crit in range(n_crit):
    
        loss = flimGANE.train_on_batch([fakeinput, irf], [A, tau1, tau2])
        gan_loss.append(loss)
        
        loss_test = flimGANE.test_on_batch([X_valid, irf_valid], [A_valid, tau1_valid, tau2_valid])
        gan_val.append(loss_test)
        
        print("Epoch #{}-{}: Generative Loss: {}".format(epoch+1, crit+1, gan_loss[-1]))
        print("Epoch #{}-{}: Generative TestLoss: {}".format(epoch+1, crit+1, gan_val[-1]))
              
    G_name = '/flimGANE_G_model_ver' + version + '_iter_' + str(epoch) + '.h5' 
    E_name = '/flimGANE_E_model_ver' + version + '_iter_' + str(epoch) + '.h5' 
    flimGANE_name = '/flimGANE_model_ver' + version + '_iter_' + str(epoch) + '.h5' 
    G.save(savepath + G_name) 
    E.save(savepath + E_name) 
    flimGANE.save(savepath + flimGANE_name)
    np.save(savepath + '/flimgane_g_loss_ver' + version + '_iter_' + str(epoch) + '.npy', gan_loss) 
    np.save(savepath + '/flimgane_g_valloss_ver' + version + '_iter_' + str(epoch) + '.npy', gan_val) 

toc = time.clock()
elapseTime = toc - tic

# Save the model and results
G_name = 'flimGANE_G_model_ver' + version + '.h5' 
E_name = 'flimGANE_E_model_ver' + version + '.h5' 
flimGANE_name = 'flimGANE_model_ver' + version + '.h5' 
G.save(savepath + G_name) 
E.save(savepath + E_name) 
flimGANE.save(savepath + flimGANE_name)
np.save(savepath + 'flimgane_g_loss_ver' + version + '.npy', gan_loss) 
np.save(savepath + 'flimgane_g_valloss_ver' + version + '.npy', gan_val) 
