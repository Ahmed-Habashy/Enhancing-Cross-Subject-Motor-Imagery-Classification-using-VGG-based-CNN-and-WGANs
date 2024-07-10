"""
Created on Sun May 26 08:17:54 2024

@author: ahmed
"""
from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
import numpy as np
from keras import backend
from keras.optimizers import RMSprop,  Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.initializers import RandomNormal
from keras.constraints import Constraint
from keras.constraints import MaxNorm
from keras.callbacks import ModelCheckpoint
import os
from os import makedirs
import tensorflow as tf
import cv2
import matplotlib.image as mpimg
import  matplotlib.pyplot as plt
from PIL import Image 

IMG_WIDTH = 32    
IMG_HEIGHT = 32
seed = 7

def create_dataset(img_folder):       
    img_data_array=[]
    class_name=[]       
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image=mpimg.imread(image_path)
            # image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image = (image - 127.5) / 127.5
            # image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    # extract the image array and class name
    (img_data, class_name) = (img_data_array,class_name)
    # Create a dictionary for all unique values for the classes
    target_dict={s: v for v, s in enumerate(np.unique(class_name))}
    target_dict
    # Convert the class_names to their respective numeric value based on the dictionary
    target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]
    x=tf.cast(np.array(img_data), tf.float32).numpy()
    y=tf.cast(list(map(int,target_val)),tf.int32).numpy()
    return x, y

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

# define the standalone critic model
def define_critic(in_shape=(IMG_WIDTH,IMG_HEIGHT,1)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = ClipConstraint(0.01)
    # define model
    model = Sequential()
    # downsample to 
    model.add(Conv2D(32, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # downsample to 
    model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # scoring, linear activation
    model.add(Flatten())
    model.add(Dense(1))
    # compile model
    opt = RMSprop(learning_rate=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

# define the standalone generator model
def define_generator(latent_dim):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # define model
    model = Sequential()
    # foundation for 8*8 image
    n_nodes = 64 * 4* 4   #128 model2
    model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4,4, 64)))
    # # upsample to
    # model.add(Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.2))
    # upsample to
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(32, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # output 28x28x1
    model.add(Conv2D(1, 4, activation='tanh', padding='same', kernel_initializer=init))
    return model

# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
    # make weights in the critic not trainable
    for layer in critic.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the critic
    model.add(critic)
    # compile model
    opt = RMSprop(learning_rate = 0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

# load images
def load_real_samples(cl_n,folder_dir):
    # load dataset
    # (trainX, trainy), (_, _) = load_data()
    (trainX, trainy) = create_dataset(folder_dir)
    # select all of the examples for a given class
    selected_ix = trainy == cl_n
    X = trainX[selected_ix]
    Y = trainy[selected_ix]
    # expand to 3d, e.g. add channels
    # X = expand_dims(X, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # # scale from [0,255] to [-1,1]
    # X = (X - 127.5) / 127.5
    return X,cl_n+1,Y

# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # select images
    X = dataset[ix]
    # generate class labels, -1 for 'real'
    y = -ones((n_samples, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels with 1.0 for 'fake'
    y = ones((n_samples, 1))
    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim,subject, cl, n_samples=100):
    # prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    # X = (X + 1) / 2.0
    # plot images
    plt.figure(figsize=(IMG_HEIGHT,IMG_WIDTH))
    for i in range(4 * 4):
        # define subplot
        plt.subplot(4, 4, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(X[i, :, :, 0]) #, cmap='gray'
    # save plot to file
    makedirs('cross/target_{} GAN_results_mix/generated_plots'.format(subject), exist_ok=True) 
    makedirs('cross/target_{} GAN_results_mix/models'.format(subject), exist_ok=True) 
    plt.savefig('cross/target_{} GAN_results_mix/generated_plots/GAN_cl{}_{}.png'.format(subject, cl, step+1))
    plt.close()
    if (step > 1000):
        # save the generator model
        g_model.save('cross/target_{} GAN_results_mix/models/GAN_cl{}_{}.h5'.format(subject, cl, step+1) )
    print('>Saved: models and plots' )

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, subject, cl):
    # plot history
    plt.plot(d1_hist, label='crit_real')
    plt.plot(d2_hist, label='crit_fake')
    plt.plot(g_hist, label='gen')
    plt.legend()
    plt.savefig('cross/target_{} GAN_results_mix/plot_line_plot_loss_cl{}.png'.format(subject,cl) )
    plt.close()

# train the generator and critic
# if all subs are i/p >>> n_epochs=40, n_batch=30  // for only one subject n_epochs=120, n_batch=5  
# // n_epochs=250, n_batch=5 >> 25%
def train(subject, cl, g_model, c_model, gan_model, dataset, latent_dim, n_epochs=40, n_batch=20, n_critic=7):   
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    print("GAN bat_per_epo",bat_per_epo )
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    print("GAN steps", n_steps)
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # lists for keeping track of loss
    c1_hist, c2_hist, g_hist = list(), list(), list()
    # manually enumerate epochs
    # for i in range(n_steps):
    for i in range(0,1800):

        # update the critic more than the generator
        c1_tmp, c2_tmp = list(), list()
        for _ in range(n_critic):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update critic model weights
            c_loss1 = c_model.train_on_batch(X_real, y_real)
            c1_tmp.append(c_loss1)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update critic model weights
            c_loss2 = c_model.train_on_batch(X_fake, y_fake)
            c2_tmp.append(c_loss2)
        # store critic loss
        c1_hist.append(mean(c1_tmp))
        c2_hist.append(mean(c2_tmp))
        # prepare points in latent space as input for the generator
        X_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = -ones((n_batch, 1))
        # update the generator via the critic's error
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        g_hist.append(g_loss)
        # summarize loss on this batch
        print('>%d, cr=%.3f, cf=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
        # evaluate the model performance every 'epoch'
        if (i+1) >= 200 and (i+1)% 100 == 0: #
            summarize_performance(i, g_model, latent_dim,subject, cl)
            epochs = i+1
    summarize_performance(i, g_model, latent_dim,subject, cl)    
    # line plots of loss
    plot_history(c1_hist, c2_hist, g_hist,subject, cl)
    return epochs

# create GAN  images    
def create_GAN_plot(examples,cl, subject):
    makedirs('cross/target_{} GAN_results_mix/GAN_dataset/CL{}'.format(subject, cl), exist_ok=True) 

    # plot images
    plt.figure(figsize=(IMG_HEIGHT, IMG_WIDTH))
    for i in range(len(examples)):
        # define subplot
        plt.axis('off')
        plt.gray()
        # plot raw pixel data
        plt.imshow(examples[i, :, :, 0], cmap='gray')

        plt.savefig('cross/target_{0} GAN_results_mix/GAN_dataset/CL{1}/cl{1}_t{2}.jpg'.format(subject, cl , i),
                    bbox_inches= 'tight', pad_inches= 0)
        Image.open('cross/target_{0} GAN_results_mix/GAN_dataset/CL{1}/cl{1}_t{2}.jpg'.format(subject, cl , i)).convert('L').save('cross/target_{0} GAN_results_mix/GAN_dataset/CL{1}/cl{1}_t{2}.jpg'.format(subject, cl , i))
        plt.close()


Drp1= 2
Drp2= 2
Drp3= 4
Drp4= 4
   

def create_cnn2(): #dropout_rate1=0.0, dropout_rate2=0.0 , momentum=0, weight_constraint=0, 
    model= Sequential()
    model.add(Dropout(Drp1/10, input_shape = (IMG_HEIGHT,IMG_WIDTH,1)) )     # dropout 1
    model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation= 'relu' 
                      , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
    model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation= 'relu' 
                      , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp2/10))                                             # dropout 2
    model.add(Conv2D(32, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   )) # kernel_regularizer=l2(0.001),
    model.add(Conv2D(32, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp2/10))                                             # dropout 3
    model.add(Conv2D(64, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp3/10))           
    model.add(Flatten())
    model.add(Dense(128, activation= 'relu' , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3) ) )  
    model.add(Dropout(Drp3/10))                                             # dropout 4
    model.add(Dense(4, activation= 'softmax' , kernel_initializer='he_uniform' )) 
    # opt = SGD(learning_rate=0.0001, momentum=0.99)
    opt = Adam(learning_rate=0.0002, beta_1 = 0.5, beta_2 = 0.8) #, beta_2 = 0.8
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    # model.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(), metrics=['accuracy'])

    return model

def create_cnn3():
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(IMG_HEIGHT,IMG_WIDTH, 1)))
    model.add(Conv2D(32, 3, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4, activation= 'softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(), metrics=['accuracy'])
    return model

def Mod_cnn2():
    model= Sequential()
    model.add(Dropout(Drp1/10, input_shape = (IMG_HEIGHT,IMG_WIDTH,1)) )     # dropout 1
    model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation= 'relu' 
                      , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
    model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation= 'relu' 
                      , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp2/10))                                             # dropout 2

    model.add(Conv2D(64, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   )) # kernel_regularizer=l2(0.001),
    model.add(Conv2D(64, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp2/10))                                             # dropout 3
    
    model.add(Conv2D(128, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   ))
    model.add(Conv2D(128, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp3/10))           
    model.add(Flatten())
    model.add(Dense(256, activation= 'relu' , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3) ) )  
    model.add(Dropout(Drp3/10))                                             # dropout 4
    model.add(Dense(4, activation= 'softmax' , kernel_initializer='he_uniform' )) 
    # opt = Adam(learning_rate=0.0002, beta_1 = 0.5, beta_2 = 0.8) #, beta_2 = 0.8
    # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    model.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(), metrics=['accuracy'])
    return model

# GAN-CNN Model training:
def model_training( x_data, y_data ,x_test, y_test,save_dir, sel_mod ,fig_title, cnn_epochs, cnn_batch_size ):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print ("Training Data= ",x_data.shape, y_data.shape ) 
    print ("Validation Data= ",x_test.shape, y_test.shape )
    print ('epochs =',cnn_epochs)
    model_tr = sel_mod
    # 1-Times generated data:
    mcg = ModelCheckpoint(save_dir, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    history = model_tr.fit(x_data, y_data, epochs= cnn_epochs , batch_size=cnn_batch_size, verbose=0 
                        ,validation_data=(x_test, y_test), callbacks=[mcg] )


    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(fig_title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.grid()
    plt.show()
    plt.close()
    
