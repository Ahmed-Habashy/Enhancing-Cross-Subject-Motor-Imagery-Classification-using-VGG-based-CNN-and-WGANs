from __future__ import print_function
import sys

sys.path.append('./gumpy')
import gumpy
import numpy as np
import utilss
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from os import makedirs
import mne

DEBUG = True
CLASS_COUNT = 2

# parameters for filtering data
FS = 250
LOWCUT = 8
HIGHCUT = 30
ANTI_DRIFT = 0.5
CUTOFF = 50.0  # freq to be removed from signal (Hz) for notch filter
Q = 30.0  # quality factor for notch filter
W0 = CUTOFF / (FS / 2)
AXIS = 0

# set random seed
SEED = 42
KFOLD = 5

# Load raw data
# Before training and testing a model, we need some data. The following code shows how to load a dataset using ``gumpy``.
# specify the location of the GrazB datasets
data_dir = './data/Graz'
subject = 'B05'  #'B01'
print ("Subject >> ", subject)
 # make folder for results
makedirs('\spectrogram_ch3,4\\sub_{}\CL1'.format(subject), exist_ok=True) 
makedirs('\spectrogram_ch3,4\\sub_{}\CL2'.format(subject), exist_ok=True) 
makedirs('\spectrogram_ch3,4\\CL1_3ch', exist_ok=True) 
makedirs('\spectrogram_ch3,4\\CL2_3ch', exist_ok=True) 

# initialize the data-structure, but do _not_ load the data yet
grazb_data = gumpy.data.GrazB(data_dir, subject,  True)

# now that the dataset is setup, we can load the data. This will be handled from within the utils function,
# which will first load the data and subsequently filter it using a notch and a bandpass filter.
# the utility function will then return the training data.
x_train, y_train = utilss.load_preprocess_data(grazb_data, True, LOWCUT,
                                              HIGHCUT, W0, Q, ANTI_DRIFT, CLASS_COUNT, CUTOFF,
                                              AXIS, FS, T= True)
# Test data:
grazb_data_test = gumpy.data.GrazB(data_dir, subject, T = False)

x_test, y_test = utilss.load_preprocess_data(grazb_data_test, True, LOWCUT,
                                              HIGHCUT, W0, Q, ANTI_DRIFT, CLASS_COUNT, CUTOFF,
                                              AXIS, FS, T = False)

def MI_4sec_data(x,y):
    # sub_data = np.rollaxis(x, 2, 1)
    # samples_win = 2*FS      # 2sec window
    # samples_str = 1125   # begining at second 4.5
    samples_win = 4*FS     # 4sec window
    samples_str = 1000   # begining at second 4
    
    MI_data = x[:, samples_str:samples_str + samples_win ,: ]
    count = 0
    for i in range( len(y) ):
        if (y[i,0] == 1 ):
            count += 1
    MI_cl1 = np.zeros(( (count) , int (3) ))
    MI_cl2 = np.zeros(( (len(y)-count) , int (3) ))
    
    MI_cl1 = MI_data[0:count, :,:]
    MI_cl2 = MI_data[count:, :,:]
    
    return MI_cl1,MI_cl2

MI_cl1, MI_cl2 = MI_4sec_data(x_train, y_train)
MI_cl1_test,MI_cl2_test = MI_4sec_data(x_test, y_test )

#%%
#================== concatenate images =========================
from PIL import Image

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height )) # 'L' for gray scale / 'RGB'
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    # dst.paste(im3, (0, im1.height+im2.height))

    return dst
#%%
from scipy import signal
# import scipy.signal 



def stft_data(X, window_size=256, draw=False, cl = 1):
    ''' 
    Short-time-Fourier transform (STFT) function
    INPUTS:  
    X - EEG data (num_trials, num_eeg_electrodes, time_bins) => (trials  x samples x channels )
    window_size - fixed # of time bins to analyze frequency components within
    stride - distance between starting position of adjacent windows
    freq - frequency range to obtain spectral power components
    OUTPUTS: 
    X_stft - STFT transformed eeg data (num_trials, num_eeg_electrodes*freq_bins,time_bins,1)
    num_freq - number of frequency bins
    num_time - number of time bins
    '''
    fs = FS
    num_trials = X.shape[0]
    num_samples= X.shape[1]
    f, t, Zxx = signal.stft(X[0,:,0], fs=fs,  nperseg=window_size, noverlap=1)
    num_freq= f.shape[0]
    num_time= t.shape[0]
    #Z_mean =  np.empty((Zxx.shape[0],Zxx.shape[1]))
    ch_stft= np.empty((int(num_trials), int(num_freq),int(num_time)))
    
    sel_channels = (0,2)  # c3,cz,c4

    for i in range(num_trials):
        for j in sel_channels:   
            f, t, Zxx = signal.stft(X[i,:,j], fs=fs,  nperseg= window_size, noverlap=1)
            ch_stft[i] = Zxx 
            #print('ch_stft.shape ',ch_stft.shape)
            plt.figure(figsize=(12,5))
            if draw==True:
                
                plt.pcolormesh(t, f, np.abs(Zxx),   cmap='jet', shading='gouraud')  # cmap='jet' / 'gray'
                plt.xlim(0,2)
                plt.ylim(8,30)
                plt.axis('off')
                # plt.title('STFT Magnitude_ch:%d' %j )
                # plt.ylabel('Frequency [Hz]')
                # plt.xlabel('Time [sec]')
                plt.savefig('D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_2b GAN\spectrogram_ch3,4\\CL{0}_3ch\cl{0} STFT_{3}_ch {1}_t{2} .png'.format(cl ,j, i, subject), bbox_inches= 'tight', pad_inches= 0)
                plt.show() 
        img26 = Image.open('\spectrogram_ch3,4\\CL{0}_3ch\cl{0} STFT_{2}_ch 0_t{1} .png' .format(cl , i, subject)) # Path to image
        img30 = Image.open('\spectrogram_ch3,4\\CL{0}_3ch\cl{0} STFT_{2}_ch 2_t{1} .png' .format(cl , i, subject)) # Path to image
        get_concat_v(img26,  img30).save('\spectrogram_ch3,4\\sub_{2}\CL{0}\cl{0} ch3_{2}_tr{1}.bmp' .format(cl , i, subject)) 
    
    return ch_stft, f , t
 
# Apply the function
cl1_stft, num_freq, num_time =stft_data(MI_cl1,256,draw=True,cl = 1)
cl2_stft, num_freq, num_time =stft_data(MI_cl2,256,draw=True,cl = 2)
cl1_stft, num_freq, num_time =stft_data(MI_cl1_test,256,draw=True,cl = 1)
cl2_stft, num_freq, num_time =stft_data(MI_cl2_test,256,draw=True,cl = 2)






