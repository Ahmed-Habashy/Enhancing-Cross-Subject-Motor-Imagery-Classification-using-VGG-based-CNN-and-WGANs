# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:15:13 2023
@author: ahmed
"""
from numpy import mean
from numpy import std
import numpy as np
from keras.models import load_model
from os import makedirs
from datetime import datetime 

subjects  = [1, 2, 3, 4, 5, 6, 7, 8, 9 ]
subjects_s  = [ 5 ]
# subjects = [ 'sub_B01', 'sub_B02' , 'sub_B03' , 'sub_B04', 'sub_B05' , 'sub_B06' , 'sub_B07' , 'sub_B08', 'sub_B09' ]   
# subjects_s = [   'sub_B08'  ]    

seed = 7
# defining the input images size    
IMG_WIDTH = 32    #224
IMG_HEIGHT = 32
 
Ad_times = 1 # 0.25 0.5 0.75
cnn_batch_size = 60
cnn_epochs = 350
sel_data = 00

from func_cnn_Wgan import *

#%%   ============================== Data Loading ============================= 
for subject in subjects_s:
    # 3-channels (C3- Cz- C4):
    # img_folder =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\IV_2a_CNN_WGAN\spectrogram\gray\3ch\sub_[{}]'.format(subject) 
    # img_folder_test =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\IV_2a_CNN_WGAN\spectrogram\gray\3ch\eval\sub_[{}]'.format(subject) 
    # 2-channels only :
    img_folder =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\IV_2a_CNN_WGAN\spectrogram\gray\2ch\sub_[{}]'.format(subject) 
    img_folder_test =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\IV_2a_CNN_WGAN\spectrogram\gray\2ch\eval\sub_[{}]'.format(subject) 
    # Dataset 2B:
    # img_folder =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_2b GAN\spectrogram\gray\{}_g'.format(subject) 
    # img_folder_test =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_2b GAN\spectrogram\gray\Test\{}_gt'.format(subject) 
   
    # 2-channels only (C3-C4) 
 
    x_tr,y_tr = create_dataset(img_folder)
    x_ev,y_ev = create_dataset(img_folder_test)
    x0 = np.concatenate((x_tr,x_ev))
    y0 = np.concatenate((y_tr,y_ev))
    # Get unique values in y
    # unique_values = np.unique(y_tr)
    # Select certain classes:
    unique_values = [1,2]

    print ('\n Subject A0',subject)

    # Create variables dynamically and print their shapes
    for val in unique_values:
        globals()[f'loso{val}'] = x_tr[y_tr == val]
        globals()[f'yloso{val}'] = y_tr[y_tr == val]
        # Select a portion of each variable
        globals()[f'loso{val}'] = globals()[f'loso{val}'][:sel_data]
        globals()[f'yloso{val}'] = globals()[f'yloso{val}'][:sel_data]
        print(f'loso{val} shape: {globals()[f"loso{val}"].shape}')
        print(f'yloso{val} shape: {globals()[f"yloso{val}"].shape}')

    
    #%% =============== Cross subject data =================================
    for sub in subjects:
        # 3-channels:
        # cr_img =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\IV_2a_CNN_WGAN\spectrogram\gray\3ch\sub_[{}]'.format(sub) 
        # cr_img_test =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\IV_2a_CNN_WGAN\spectrogram\gray\3ch\eval\sub_[{}]'.format(sub) 
        # 2-channels only:
        cr_img =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\IV_2a_CNN_WGAN\spectrogram\gray\2ch\sub_[{}]'.format(sub) 
        cr_img_test =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\IV_2a_CNN_WGAN\spectrogram\gray\2ch\eval\sub_[{}]'.format(sub)       
        # Dataset 2B:
        # cr_img =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_2b GAN\spectrogram\gray\{}_g'.format(sub) 
        # cr_img_test =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_2b GAN\spectrogram\gray\Test\{}_gt'.format(sub)   
        #     
        if (sub==subject):
            continue

        print ("\n Cross_sub",sub)
        for val in unique_values:
            loso_var = globals()[f'loso{val}']
            yloso_var = globals()[f'yloso{val}']
        
            # Load and concatenate training data
            dataset1, cl, y_cr = load_real_samples(val, cr_img)
            loso_var = np.concatenate((loso_var, dataset1), axis=0)
            yloso_var = np.concatenate((yloso_var, y_cr), axis=0)
            print(f"cl{cl} data =", loso_var.shape)
        
            # Load and concatenate test data
            dataset2, cl, y_cr = load_real_samples(val, cr_img_test)
            loso_var = np.concatenate((loso_var, dataset2), axis=0)
            yloso_var = np.concatenate((yloso_var, y_cr), axis=0)
            print("load eval data --- ",f"cl{cl} data =", loso_var.shape)
            # ========= Update globals ===========
            globals()[f'loso{val}'] = loso_var
            globals()[f'yloso{val}'] = yloso_var   
              
#%% ============================= GAN Training =====================================
    # # load image data cl 1 
    latent_dim = 100
    # critic = define_critic()
    # generator = define_generator(latent_dim)
    # gan_model = define_gan(generator, critic)
    # # train model
    # cl = 1
    # print (f"Wgan class {cl} Training data:{loso1.shape} " )
    # start = datetime.now()
    # epochs1 = train(subject,cl, generator, critic, gan_model, loso1, latent_dim)
    # duration = datetime.now() - start
    # print("Training completed in time: ", duration)
    # # load image data cl 2 
    # critic = define_critic()
    # generator = define_generator(latent_dim)
    # gan_model = define_gan(generator, critic)
    # # train model
    # cl = 2
    # print (f"Wgan class {cl} Training data:{loso1.shape} " )
    # start = datetime.now()
    # epochs2 = train(subject,cl,generator, critic, gan_model, loso2, latent_dim)  
    # duration = datetime.now() - start
    # print("Training completed in time: ", duration)
#%%========================== Generating GAN Images =========================== 
    # epochs1 = 1600
    # epochs2 = 1800
    # cl=1   
    # print("\n Generating atrifitial images class  ", cl)
    # # load model
    # model1 = load_model('cross/target_{} GAN_results_mix/models/GAN_cl{}_{}.h5'.format(subject, cl, epochs1))
    # # generate images
    # latent_points1 = generate_latent_points(latent_dim,  int ((len(loso1)) * Ad_times) )   
    # # generate images
    # X_gan_lat1 = model1.predict(latent_points1)    
    # # plot the result
    # create_GAN_plot(X_gan_lat1,1,subject)
    
    # cl=2
    # tf.random.set_seed(seed)
    # np.random.seed(seed)
    # print("\n Generating atrifitial images class  ", cl)

    # model2 = load_model('cross/target_{} GAN_results_mix/models/GAN_cl{}_{}.h5'.format(subject, cl, epochs2))
    # # generate images
    # latent_points2 = generate_latent_points(latent_dim, int ((len(loso2)) * Ad_times))
    # # generate images
    # X_gan_lat2 = model2.predict(latent_points2)    
    # # plot the result
    # create_GAN_plot(X_gan_lat2,2,subject)
    #%% ============================= Collect all loso and yloso arrays ============================

    loso_list = [globals()[f'loso{val}'] for val in unique_values]
    yloso_list = [globals()[f'yloso{val}'] for val in unique_values]
    
    # Concatenate all loso and yloso arrays
    x_tr = np.concatenate(loso_list, axis=0)
    y_tr = np.concatenate(yloso_list, axis=0)
    
    # Print the shapes of the concatenated arrays
    print(f'x_tr without WGAN shape: {x_tr.shape}')
    print(f'y_tr without WGAN shape: {y_tr.shape}')
    
    # Adject EV data for 2-classes problem
    x_ev1,y_ev1 = x_ev[y_ev==1], y_ev[y_ev==1]
    x_ev2,y_ev2 = x_ev[y_ev==2], y_ev[y_ev==2]  
    x_ev = np.concatenate((x_ev1, x_ev2), axis=0)   
    y_ev = np.concatenate((y_ev1, y_ev2), axis=0)   
    # x_ev1,y_ev1 = x0[y0==1], y0[y0==1]
    # x_ev2,y_ev2 = x0[y0==2], y0[y0==2]  
    # x_ev = np.concatenate((x_ev1, x_ev2), axis=0)   
    # y_ev = np.concatenate((y_ev1, y_ev2), axis=0)    
    #%% ========== Cross GAN Data ========
    print ("\n WGAN sub:",subject)
    GAN_data= r'cross/target_{0} GAN_results_mix/GAN_dataset'.format(subject)      
    x_gan, y_gan =  create_dataset(GAN_data)
    x_tr = np.concatenate((x_tr, x_gan), axis=0)
    y_tr = np.concatenate((y_tr, y_gan+1), axis=0)
    print(f'x_tr after WGAN shape: {x_tr.shape}')
    print(f'y_tr after WGAN shape: {y_tr.shape}')
#%% ================================== CNN training =================================
    start = datetime.now()
    # model_training(x_tr, y_tr, x_ev, y_ev,'cross/cnn_models/2ch/sub_{}_zero_cr_modCNN_.h5'.format(subject),
    #                 Mod_cnn2(), '{}_CNN2 Model accuracy'.format(subject), cnn_epochs, cnn_batch_size)     
    model_training(x_tr, y_tr, x_ev, y_ev,'cross/cnn_models/2ch/sub_{}_zero_cr_wgan_modCNN_.h5'.format(subject),
                    Mod_cnn2(), '{}_CNN2 Model accuracy'.format(subject), cnn_epochs, cnn_batch_size)     

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
#%% test
def average_acc ():
    acc_scores = list()     
    for sub in subjects:
        # 3 Ch or 2 ch:
        img_folder_test =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\IV_2a_CNN_WGAN\spectrogram\gray\2ch\eval\sub_[{}]'.format(sub) 
             
        x_eval,y_eval = create_dataset(img_folder_test)
        
        # model = load_model('cross/cnn_models/2ch/sub_{}_zero_cr_CNN3.h5'.format( sub))  # "Paper 3 modified CNN + WGAN" 
        model = load_model('cross/cnn_models/2ch/sub_{}_zero_cr_wgan_modCNN.h5'.format( sub))  # "Paper 3 modified CNN + WGAN" 
        
        # Adject EV data for 2-classes problem
        x_eval1,y_eval1 = x_eval[y_eval==1], y_eval[y_eval==1]
        x_eval2,y_eval2 = x_eval[y_eval==2], y_eval[y_eval==2]  
        x_eval = np.concatenate((x_eval1, x_eval2), axis=0)   
        y_eval = np.concatenate((y_eval1, y_eval2), axis=0)  
        _, test_acc= model.evaluate(x_eval,y_eval,verbose=0)
        print('Test {}: '.format( sub),test_acc)
        acc_scores.append(test_acc)
    print('\n >>>> Accuracy: mean={} std={}, n={}' .format ( mean(acc_scores)*100
                                                                        , std(acc_scores)*100, len(acc_scores)))
average_acc()

# model = load_model('sub_ind/GAN_CNN/{}_5%_GAN_2_CNN3_{}.h5'.format( subject,   nfolds))
# _, test_acc= model.evaluate(x_ev, y_ev,verbose=0)
# print('CNN3 Test: ',nfolds,' fold Accuracy',test_acc)

