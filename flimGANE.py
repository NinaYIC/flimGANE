"""
Editor: Nina Chen
Final version
Main code for flimGANE training and testing for FLIM
11/10/2021

"""

#%% Import required modules
import os
import numpy as np                    
import pandas as pd                   
import random                         
import itertools                    
import matplotlib.pyplot as plt          
from skimage import io
from keras import backend as K        
from keras.models import Model, load_model  
from keras.layers import Input, Dense, Lambda, Concatenate, Add
from keras.layers import Conv1D, Reshape, AveragePooling1D, Flatten
from keras.optimizers import Adam, RMSprop
from mpl_toolkits.axes_grid1 import make_axes_locatable

workdir = os.getcwd() 
tm = 50               # The range of decay histogram (e.g., 50 ns)
res = 256             # Number of bins in decay histogram (e.g., 256 bins)
version = '_1'
savepath = workdir + "/Results/"

#%% Define the functions to be utilized
def generate_decay_histogram(IRF, A, tau1, tau2):
    '''
    This function is to simulate a single time decay histogram.
    Input:
        IRF         : Instrument Response Function
        A           : Fraction of shorter lifetime
        tau1        : short lifetime
        tau2        : long lifetime
    Output:
        simGroundTruthDecay : Normalized ground truth decay curve after convolution
    '''
    t = np.linspace(0, tm, res)
    component1 = A*np.exp(-t/tau1)
    component2 = (1-A)*np.exp(-t/tau2)
    exponentialDecay = component1+component2
    simGroundTruthDecay = np.convolve(IRF, exponentialDecay)
    simGroundTruthDecay = simGroundTruthDecay / max(simGroundTruthDecay)   
    return simGroundTruthDecay[:res]

def generate_data(Xtrain, IRF, FLIMA, FLIMTau1, FLIMTau2, n_samples=100000, display=False):
    '''
    This function is to generate the ground-truth and simulated decay curve/histogram.
    Input:
        Xtrain      : Simulated decay histogram from MC simulation
        IRF         : Instrument Response Function
        FLIMA       : Fraction of shorter lifetime
        FLIMtau1    : short lifetime
        FLIMtau2    : long lifetime
        n_samples   : number of samples
        display     : display the string to show the current status or not
    Output:
        real        : Ground-truth decay curve (clear inputs)
        fake        : Simulated decay histogram (messy inputs)
        irf         : Instrument Response Function
        A           : Fraction of shorter lifetime
        tau1        : short lifetime
        tau2        : long lifetime
    '''
    idx = random.sample(list(range(Xtrain.shape[0])),  n_samples)  
    A = FLIMA[idx]
    tau1 = FLIMTau1[idx]
    tau2 = FLIMTau2[idx]
    irf, real, fake = np.zeros([n_samples, res]), np.zeros([n_samples, res]), np.zeros([n_samples, res])
    fake = Xtrain[idx, :]
    irf = IRF[idx, :]
    for ind in range(n_samples):
        real[ind, :] = generate_decay_histogram(irf[ind, :], A[ind], tau1[ind], tau2[ind])   
        if ind % (n_samples/10) == 0 and ind != 0 and display:
            print(str(ind) + ' Samples Done!')    
    if display:
        print(str(n_samples) + ' Samples Done!')    
    return real, fake, irf, A, tau1, tau2

def generate_training_data(Xtrain, IRF, FLIMA, FLIMTau1, FLIMTau2, n_samples=100000):
    '''
    This function is to generate the dataset for generative model training
    Input:
        Xtrain      : Simulated decay histogram
        IRF         : Instrument Response Function
        FLIMA       : Fraction of shorter lifetime
        FLIMtau1    : short lifetime
        FLIMtau2    : long lifetime
        n_samples   : number of samples
    Output:
        real        : Ground-truth decay curve (clear inputs)
        fake        : Simulated decay histogram (messy inputs)
        real_label  : Label for ground-truth decay histogram
        fake_label  : Label for simulated decay hiostogram
        irf         : Instrument Response Function
        A           : Fraction of shorter lifetime
        tau1        : short lifetime
        tau2        : long lifetime
    '''
    real, fake, irf, A, tau1, tau2 = generate_data(Xtrain, IRF, FLIMA, FLIMTau1, 
                                                   FLIMTau2, n_samples=n_samples, display=False)
    real_label = -np.ones((real.shape[0], 1))
    fake_label  = np.ones((real.shape[0], 1))
    return real, fake, real_label, fake_label, irf, A, tau1, tau2

def generate_flimGANEtraining_data(Xtrain, IRF, FLIMA, FLIMTau1, FLIMTau2, n_samples=100000):
    '''
    This function is to generate the dataset for flimGANE G+E model training
    Input:
        Xtrain      : Simulated decay histogram
        IRF         : Instrument Response Function
        FLIMA       : Fraction of shorter lifetime
        FLIMtau1    : short lifetime
        FLIMtau2    : long lifetime
        n_samples   : number of samples
    Output:
        fake        : Simulated decay histogram (messy inputs)
        irf         : Instrument Response Function
        A           : Fraction of shorter lifetime
        tau1        : short lifetime
        tau2        : long lifetime
    '''
    real, fake, irf, A, tau1, tau2 = generate_data(Xtrain, IRF, FLIMA, FLIMTau1, 
                                                   FLIMTau2, n_samples=n_samples, display=False)
    return fake, irf, A, tau1, tau2

def divide(tensors):
    '''
    This function is utilized to implement "Normalization" mathematical operation.
    '''
    output = tensors
    outputmax = K.max(output, axis=1, keepdims=True) 
    import tensorflow as tf
    finalout = tf.math.divide(output, outputmax)     
    return finalout

def wasserstein_loss(y_true, y_pred):
    '''
    This function defines the wasserstein loss for model training.
    Input:
        y_true : ground truth 
        y_pred : model prediction
    '''
    return K.mean(y_true * y_pred) 

def get_generative(G_in_decay, G_in_irf):
    '''
    This function generate the generative model.
    Input:
        G_in_decay : G input tensor for decay histogram
        G_in_irf   : G input tensor for irf
    Output:
        G          : Generative model
        G_out      : G output tensor
    '''
    G_in_decay_reshape = Reshape((res, 1))(G_in_decay)
    G_in_irf_reshape   = Reshape((res, 1))(G_in_irf)
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
    G_out = Dense(res, activation='tanh')(G_out)
    G_out = Add()([G_in_decay, G_out])
    G_out = Lambda(divide, output_shape=(res,))(G_out)
    G = Model(inputs=[G_in_decay, G_in_irf], outputs=G_out)
    G.compile(loss='mse', optimizer='adam')
    return G, G_out

def get_discriminative(D_in_decay):
    '''
    This function generate the discriminative model.
    Input:
        D_in_decay : D input tensor for decay histogram
    Output:
        D          : Discriminative model
        D_out      : D output tensor
    '''
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
    '''
    This function generate the estimative model.
    Input:
        E_in_decay : E input tensor for decay histogram
        E_in_irf   : E input tensor for irf
    Output:
        E          : Estimatve model
        E_out      : E output tensor
    '''
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
    '''
    This function is ustilized to set the trainability of the model.
    '''
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def make_gan(GAN_in_decay, GAN_in_irf, G, D):
    '''
    This function generate the GAN model.
    Input:
        GAN_in_decay   : GAN input tensor for decay histogram
        GAN_in_irf     : GAN input tensor for irf
        G              : generative model
        D              : discrimative model
    Output:
        GAN            : GAN model
        GAN_out        : GAN output tensor
        generator_model: GAN model with additional output (G-output)
    '''
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

def make_flimGANE(GAN_in_decay, GAN_in_irf, G, E):
    '''
    This function generate the flimGANE model.
    Input:
        GAN_in_decay   : GAN input tensor for decay histogram
        GAN_in_irf     : GAN input tensor for irf
        G              : generative model
        E              : estimative model
    Output:
        flimGANE       : GAN model
        flimGANE_out   : GAN output tensor
    '''
    set_trainability(E, True)
    set_trainability(G, False)
    x = G([GAN_in_decay, GAN_in_irf])
    flimGANE_out = E([x, GAN_in_irf])
    flimGANE = Model([GAN_in_decay, GAN_in_irf], flimGANE_out)
    flimGANE.compile(loss='mse', optimizer='adam')
    return flimGANE, flimGANE_out


# -------------------------------
#%% Main code start here
# -------------------------------

#%% Start with generative model training
# Load the data set 
sim_filename = '/Example_SimDataset.pkl'
dataset = pd.read_pickle(workdir + sim_filename)
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
irfFileName = '/Example_IRF.tif'
irf = io.imread(workdir + irfFileName)
# Use the same simulation parameters as you did for the previous generative model training
tau1s = np.linspace(0.3, 6.0, 58)
tau2s = [0.5]
alphas = [0.99, 1.00]
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
            histograms[it, :] = generate_decay_histogram(irf, alpha, tau1, tau2)
            IRF[it, :] = irf
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

for epoch in range(epochs):
    print("Epoch #{}.......".format(epoch+1))    
    e_loss.append(E.train_on_batch([histograms, IRF], [FLIMA, FLIMtau1, FLIMtau2]))
    if verbose and (epoch+1) % v_freq == 0:  
        print("Epoch #{}: Discriminative Loss: {}".format(epoch+1, e_loss[-1]))

# Assign the version to be saved and save the results
E_name = 'E_model_ver' + version + '.h5' 
E.save(savepath + E_name) 
np.save(savepath + 'e_loss_ver' + version + '.npy', e_loss)

#%% Start flimGANE combinative training
# First, load the dataset if you haven't
# sim_filename = '/Example_SimDataset.pkl'
# dataset = pd.read_pickle(workdir + sim_filename)
Xtrain = np.reshape(dataset['TimeDecayHistogram'][0], (-1, res))
IRF = np.reshape(dataset['IRF'][0], (-1, res))
FLIMA = np.reshape(dataset['FLIM_A'][0], (-1, 1))
FLIMTau1 = np.reshape(dataset['FLIM_tau1'][0], (-1, 1))
FLIMTau2 = np.reshape(dataset['FLIM_tau2'][0], (-1, 1))
for i in range(np.shape(IRF)[0]):
    IRF[i, :] = IRF[i, :]/np.max(IRF[i, :])

# Load the model
#G = load_model(workdir + '/WGAN_G_model_ver_1.h5', custom_objects=dict(wasserstein_loss=wasserstein_loss))
#E = load_model(workdir + '/E_model_ver_1.h5')
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
              

# Save the model and results
G_name = 'flimGANE_G_model_ver' + version + '.h5' 
E_name = 'flimGANE_E_model_ver' + version + '.h5' 
flimGANE_name = 'flimGANE_model_ver' + version + '.h5' 
G.save(savepath + G_name) 
E.save(savepath + E_name) 
flimGANE.save(savepath + flimGANE_name)
np.save(savepath + 'flimgane_g_loss_ver' + version + '.npy', gan_loss) 
np.save(savepath + 'flimgane_g_valloss_ver' + version + '.npy', gan_val) 


#%% flimGANE prediction
# Load the dataset
# ================================================================
irffilename = workdir + "/Example_IRF.tif"
decayfilename = workdir + "/Example_Decay.pkl"
flimGANEfilename = workdir + "/flimGANE_model_ver_1.h5"
# ================================================================

# IRF
irf = io.imread(irffilename)
    
# Decay curve
dataset = pd.read_pickle(decayfilename) 
decay_ = dataset['TimeDecayHistogram'][0]         
intensity_ = np.sum(decay_, axis=-1)

# flimGANE model
flimGANE = load_model(flimGANEfilename, 
                      custom_objects=dict(wasserstein_loss=wasserstein_loss)) 

# flimGANE prediction
flimGANE_FLIM = np.zeros((14, 47))
for dimx, dimy in itertools.product(range(14), range(47)):
    if intensity_[dimx, dimy] > 0:   
        IRF  = np.reshape(irf, (-1, 256))
        data = decay_[dimx, dimy, :]
        data = data / max(data)
        data = np.reshape(data, (1, -1))
        
        flimGANE_prediction = flimGANE.predict([data, IRF])
        flimGANE_FLIM[dimx, dimy] = flimGANE_prediction[0]*flimGANE_prediction[1] + (1-flimGANE_prediction[0])*flimGANE_prediction[2]

# Visualize the flimGANE result image
fig, ax  = plt.subplots(figsize=(9, 9))
image    = ax.imshow(flimGANE_FLIM, cmap=plt.get_cmap('CMRmap'))
image.set_clim([0, 6])
plt.xticks([])
plt.yticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='10%', pad=0.1)
fig.colorbar(image, cax=cax, orientation="vertical")
plt.tight_layout()
