
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014001
# from moabb.paradigms import FilterBankMotorImagery, LeftRightImagery, MotorImagery
from moabb.paradigms import  MotorImagery
import numpy as np
from os import makedirs

# Load dataset
dataset = BNCI2014001()
subjects  = [1, 2, 3, 4, 5, 6, 7, 8, 9 ]
subject  = [ 1 ]

FS = 250


paradigm = MotorImagery(n_classes=4, channels=["C3","C4"], fmin= 8.0, fmax= 30 )  # channels=["C3", "Cz", "C4"], baseline= (4,5)
# print(paradigm.__doc__)
# Print data information
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subject)
print(X.shape)
print(np.unique(y))
print(metadata.head())
# Ensure each trial has 1000 samples
# X = X[:, :, :1000]
# Split data based on sessions in metadata
# Identify training and evaluation sessions from metadata
train_sessions = metadata[metadata['session'] == 'session_T'].index
eval_sessions = metadata[metadata['session'] == 'session_E'].index

X_train = X[train_sessions]
y_train = y[train_sessions]
X_test = X[eval_sessions]
y_test = y[eval_sessions]

print(f'Training set shape: {X_train.shape}')
print(f'Testing set shape: {X_test.shape}')
labels = np.unique(y)
X_groups = {label: X_train[y_train == label] for label in labels}
X_gt = {label: X_test[y_test == label] for label in labels}

# Print shapes of the groups to verify the split
for label in labels:
    print(f"Group {label} data: {X_groups[label].shape}")

# Choose a specific group 
X1 = X_groups["feet"]
X2 = X_groups["left_hand"]
X3 = X_groups["right_hand"]
X4 = X_groups["tongue"]

X1_t = X_gt["feet"]
X2_t = X_gt["left_hand"]
X3_t = X_gt["right_hand"]
X4_t = X_gt["tongue"]

#%%================== concatenate images =========================
from PIL import Image
    # 3 Channels:
# def get_concat_v(im1, im2, im3):
    # dst = Image.new('L', (im1.width, im1.height + im2.height + im3.height)) # 'L' for gray scale / 'RGB'
    # 2 Channels:
def get_concat_v(im1, im2):
    dst = Image.new('L', (im1.width, im1.height + im2.height )) # 'L' for gray scale / 'RGB'
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    # dst.paste(im3, (0, im1.height+im2.height))  # for 3 ch 

    return dst
#%%
from scipy import signal
# import scipy.signal 

# make folder for results
for i in range(4):
    makedirs('spectrogram/gray/2ch//sub_{}/CL{}'.format(subject,i+1), exist_ok=True) 
    makedirs('spectrogram/gray/2ch//CL{}_2ch'.format(i+1), exist_ok=True) 


def stft_data(X, window_size=256, draw=False, cl = 1):
    ''' 
    Short-time-Fourier transform (STFT) function
    INPUTS:  
    X - EEG data (num_trials, num_eeg_electrodes, time_bins) 
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
    f, t, Zxx = signal.stft(X[0,0,:], fs=fs,  nperseg=window_size, noverlap=1)
    num_freq= f.shape[0]
    num_time= t.shape[0]
    print ("freq_bins:",num_freq,"time_bins:",num_time)
    ch_stft= np.empty((int(num_trials), int(num_freq),int(num_time)))
    
    for i in range(num_trials):
        for ch in range(X.shape[1]):   
            f, t, Zxx = signal.stft(X[i,ch,:], fs=fs,  nperseg= window_size, noverlap=1)
            ch_stft[i] = Zxx 
            #print('ch_stft.shape ',ch_stft.shape)
            plt.figure(figsize=(12,5))
            if draw==True:
                
                plt.pcolormesh(t, f, np.abs(Zxx),   cmap='gray', shading='gouraud')  # cmap='jet' / 'gray'
                plt.xlim(0,2)
                plt.ylim(8,30)
                plt.axis('off')
                plt.savefig('spectrogram/gray/2ch//CL{0}_2ch/cl{0} STFT_{3}_ch {1}_t{2} .png'.format(cl ,ch, i, subject), bbox_inches= 'tight', pad_inches= 0)
                plt.show() 
        img26 = Image.open('spectrogram/gray/2ch//CL{0}_2ch/cl{0} STFT_{2}_ch 0_t{1} .png' .format(cl , i, subject)) # Path to image
        img28 = Image.open('spectrogram/gray/2ch//CL{0}_2ch/cl{0} STFT_{2}_ch 1_t{1} .png' .format(cl , i, subject)) # Path to image
        # for 3 ch (c3, cz, c4)
        # img30 = Image.open('spectrogram/gray/2ch//CL{0}_3ch/cl{0} STFT_{2}_ch 2_t{1} .png' .format(cl , i, subject)) # Path to image
        # get_concat_v(img26, img28, img30).save('spectrogram/gray/3ch//sub_{2}/CL{0}/cl{0} ch2_{2}_tr{1}.bmp' .format(cl , i, subject)) 
        # 2 ch only: 
        get_concat_v(img26, img28).save('spectrogram/gray/2ch//sub_{2}/CL{0}/cl{0} ch2_{2}_tr{1}.bmp' .format(cl , i, subject)) 
   
    return ch_stft, f , t

# Apply the function
cl1_stft, num_freq, num_time =stft_data(X1,256,draw=True,cl = 1)
cl2_stft, num_freq, num_time =stft_data(X2,256,draw=True,cl = 2)
cl3_stft, num_freq, num_time =stft_data(X3,256,draw=True,cl = 3)
cl4_stft, num_freq, num_time =stft_data(X4,256,draw=True,cl = 4)