# -*- coding: utf-8 -*-
"""
A physics-informed data-driven low order model for the wind velocity deficit at the wake of isolated buildings.

Prepared by
Dimitrios K. Fytanidis, Romit Maulik, Ramesh Balakrishnan, and Rao Kotamarthi
Argonne National Laboratory, Lemont, IL


"""



import os
import sys

import tensorflow as tf
tf.random.set_seed(10)
from tensorflow.keras import Model
import numpy as np
np.random.seed(10)
import math

import pkgutil
from io import StringIO

# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
# tf.keras.backend.set_floatx('float64')

# Plotting
import matplotlib.pyplot as plt

# loading of the data
import numpy as np


import time
#import PILOWFlogo

# Coefficient of determination
def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
    SS_res =  np.sum(np.square( y_true-y_pred )) 
    SS_tot = np.sum(np.square( y_true - np.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + 2.22044604925e-16) )

#Build the model which does basic map of inputs to targets
class regression_model(Model):
    # Order of inputs - H, W, L, x, y, z - as column inputs 
    # Last column of data is f (target)
    def __init__(self, data):
        super(regression_model, self).__init__()

        # Set up the data for training
        self.num_samples = np.shape(data)[0]

        # Not needed now
        # self.preproc_pipeline = Pipeline([('stdscaler', StandardScaler())])
        # data[:,-1] = self.preproc_pipeline.fit_transform(data[:,-1].reshape(-1,1))[:,0]

        idx = np.arange(self.num_samples)
        np.random.shuffle(idx)
        shuffled_data = data[idx]

        self.training_data = shuffled_data[:int(0.7*self.num_samples)]
        self.validation_data = shuffled_data[int(0.7*self.num_samples):int(0.8*self.num_samples)]
        self.testing_data = shuffled_data[int(0.8*self.num_samples):]

        self.ntrain = self.training_data.shape[0]
        self.nvalid = self.validation_data.shape[0]
        self.ntest = self.testing_data.shape[0]

        # Define NN architecture - Perera component
        self.alpha_net_0=tf.keras.layers.Dense(30,input_shape=(3,),activation='tanh')
        self.alpha_net_1=tf.keras.layers.Dense(30,activation='tanh')
        self.alpha_net_2=tf.keras.layers.Dense(1,activation='linear')

        self.Dz_net_0=tf.keras.layers.Dense(30,input_shape=(6,),activation='tanh')
        self.Dz_net_1=tf.keras.layers.Dense(30,activation='tanh')
        self.Dz_net_2=tf.keras.layers.Dense(1,activation='linear') # Predict in log space

        self.Dy_net_0=tf.keras.layers.Dense(30,input_shape=(6,),activation='tanh')
        self.Dy_net_1=tf.keras.layers.Dense(30,activation='tanh')
        self.Dy_net_2=tf.keras.layers.Dense(1,activation='linear') # Predict in log space

        self.x0_net_0=tf.keras.layers.Dense(30,input_shape=(3,),activation='tanh')
        self.x0_net_1=tf.keras.layers.Dense(30,activation='tanh')
        self.x0_net_2=tf.keras.layers.Dense(1,activation='linear')

        # Define NN architecture - correction component
        self.yv_net_0=tf.keras.layers.Dense(30,input_shape=(3,),activation='tanh')
        self.yv_net_1=tf.keras.layers.Dense(30,activation='tanh')
        self.yv_net_2=tf.keras.layers.Dense(1,activation='linear')        

        self.gamma_net_0=tf.keras.layers.Dense(30,input_shape=(3,),activation='tanh')
        self.gamma_net_1=tf.keras.layers.Dense(30,activation='tanh')
        self.gamma_net_2=tf.keras.layers.Dense(1,activation='linear')        

        self.h_net_0=tf.keras.layers.Dense(30,input_shape=(3,),activation='tanh')
        self.h_net_1=tf.keras.layers.Dense(30,activation='tanh')
        self.h_net_2=tf.keras.layers.Dense(1,activation='linear')        

        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Running the model (eager mode - use tf.gather/gather_nd for @tf.function decorator)
    def call(self,X):
        # Order of inputs - H, W, L, x, y, z - as column inputs 
        chi = X[:,3:4]/X[:,0:1]
        yi = X[:,4:5]/X[:,0:1]
        zi = X[:,5:6]/X[:,0:1]

        net_inputs = X[:,:3]

        hh = self.alpha_net_0(net_inputs)
        hh = self.alpha_net_1(hh)
        alpha = self.alpha_net_2(hh)

        hh = self.x0_net_0(net_inputs)
        hh = self.x0_net_1(hh)
        x0 = self.x0_net_2(hh)
     
        net_inputs = X[:,:6]

        hh = self.Dz_net_0(net_inputs)
        hh = self.Dz_net_1(hh)
        logDz = self.Dz_net_2(hh)
        Dz = tf.math.exp(logDz)

        hh = self.Dy_net_0(net_inputs)
        hh = self.Dy_net_1(hh)
        logDy = self.Dy_net_2(hh)
        Dy = tf.math.exp(logDy)

        # Core Perera operations
        lamz = tf.math.abs(tf.math.pow(Dz*(chi-x0),0.5))+K.epsilon()
        lamy = tf.math.abs(tf.math.pow(Dy*(chi-x0),0.5))+K.epsilon()

        ksi = zi/(lamz)
        eta = yi/(lamy)              

        g = ksi/2.0*tf.math.exp(-ksi*ksi/4.0)
        
        h = 1/2/tf.math.sqrt(np.pi)*tf.math.exp(-eta*eta/4.0)
        f = (alpha*(X[:,1:2]/(lamy))*((X[:,0:1]/lamz)**2)*g*h)

        # Correction component
        net_inputs = X[:,:3]

        # yv
        hh = self.yv_net_0(net_inputs)
        hh = self.yv_net_1(hh)
        yv = tf.exp(self.yv_net_2(hh))

        # Gamma
        hh = self.gamma_net_0(net_inputs)
        hh = self.gamma_net_1(hh)
        gamma = self.gamma_net_2(hh)

        # small h
        hh = self.h_net_0(net_inputs)
        hh = self.h_net_1(hh)
        small_h = tf.exp(self.h_net_2(hh))

        # calculate correction 
        yp = yv - yi
        zi = tf.cast(zi,dtype='float32')
        al = tf.math.pow(yp,2)+tf.math.pow(small_h,2)+tf.math.pow(zi,2)
        #al = tf.math.pow(yp,2)+small_h+tf.math.pow(zi,2)

        fprime = tf.math.minimum(gamma*small_h*yp*chi/(tf.math.pow(al,2)-tf.math.pow(2*zi*small_h,2)+K.epsilon()),2.0)
        
        return tf.math.exp(f+fprime)
    
    # Regular MSE
    def get_loss(self,X,Y):
        op = self.call(X)

        return tf.reduce_mean(tf.math.square(op-Y))

    # get gradients - regular
    def get_grad(self,X,Y):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(X,Y)
            g = tape.gradient(L, self.trainable_variables)
        return g
    
    # perform gradient descent - regular
    def network_learn(self,X,Y):
        g = self.get_grad(X,Y)
        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # Train the model
    def train_model(self):
        plot_iter = 0
        stop_iter = 0
        patience = 100
        best_valid_loss = np.inf

        self.num_batches = 40
        self.train_batch_size = int(self.ntrain/self.num_batches)
        self.valid_batch_size = int(self.nvalid/self.num_batches)

        
        for i in range(4000):
            # Training loss
            print('Training iteration:',i)
            
            for batch in range(self.num_batches):
                input_batch = self.training_data[batch*self.train_batch_size:(batch+1)*self.train_batch_size,:-1]
                output_batch = self.training_data[batch*self.train_batch_size:(batch+1)*self.train_batch_size,-1].reshape(-1,1)
                self.network_learn(input_batch,output_batch)

            # Validation loss
            valid_loss = 0.0
            valid_r2 = 0.0

            for batch in range(self.num_batches):
                input_batch = self.validation_data[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size,:-1]
                output_batch = self.validation_data[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size,-1].reshape(-1,1)

                valid_loss = valid_loss + self.get_loss(input_batch,output_batch).numpy()
                predictions = self.call(input_batch)
                valid_r2 = valid_r2 + coeff_determination(predictions,output_batch)

            valid_r2 = valid_r2/self.nvalid
            valid_loss = valid_loss/self.nvalid

            # Check early stopping criteria
            if valid_loss < best_valid_loss:
                
                print('Improved validation loss from:',best_valid_loss,' to:', valid_loss)
                print('Validation R2:',valid_r2)
                
                best_valid_loss = valid_loss

                self.save_weights('./checkpoints/my_checkpoint')
                
                stop_iter = 0
            else:
                print('Validation loss (no improvement):',valid_loss)
                print('Validation R2:',valid_r2)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                break
                

    # Load weights
    def restore_model(self):
        checkpoint_dir = os.path.join(os.path.dirname(sys.modules['dw_tap'].__file__),
                                      'anl-lom-models/checkpoints/my_checkpoint.index')
        print("checkpoint_dir:", checkpoint_dir)
        self.load_weights(checkpoint_dir) # Load pretrained model

    # Do some testing
    def test_model(self):
        # Restore from checkpoint
        self.restore_model()

        # Check accuracy on test
        input_test = self.testing_data[:,:-1]
        output_test = self.testing_data[:,-1].reshape(-1,1)
        predictions = self.call(input_test)

        print('Test loss:',self.get_loss(input_test,output_test).numpy()/input_test.shape[0])

        output_test = (output_test)
        predictions = (predictions)

        plt.figure()
        plt.title('Scatter accuracy')
        plt.scatter(output_test[:,0],predictions[:,0],facecolor="white", edgecolor="black",label='Predicted')
        plt.plot(output_test[:,0],output_test[:,0],color="black",label='True',zorder=10)
        plt.legend()
        plt.savefig('fig100.png')

        plt.show()

        return None

    # Do some testing
    def make_predictions(self,input_data):
        
        print("input_data:", input_data)
        # Restore from checkpoint
        self.restore_model()

        # Predict for new data
        predictions = self.call(input_data)
        predictions = np.log(predictions)

        hh = self.Dz_net_0(input_data)
        logDz = self.Dz_net_1(hh)
        Dz = tf.math.exp(logDz)

        hh = self.Dy_net_0(input_data)
        logDy = self.Dy_net_1(hh)
        Dy = tf.math.exp(logDy)


        np.nan_to_num(predictions, nan=0.)
        
        
        return predictions#, Dz.numpy(), Dy.numpy()

def loadMLmodel():

    '''
    do something
    return data as (num_points,7) array

    '''
    # Order of inputs - H, W, L, x, y, z - as column inputs 
    # Last column of data is f (target)
    #  x,   y,  z,  H, Lx, Ly, U, V, W, f, fg, dUdz, k, tau
    #  0,   1,  2,  3,  4,  5, 6, 7, 8, 9, 10,   11, 12, 13  
    
    #data1 = np.loadtxt(filename)
    txt = StringIO(pkgutil.get_data('dw_tap', 'anl-lom-models/1x1x1-2-3-1x2x1i-cheb.dat').decode('UTF-8'))
    data1 = np.loadtxt(txt)
    
    data  = np.column_stack((data1[:,3], data1[:,5], data1[:,4], data1[:,0], data1[:,1], data1[:,2], data1[:,10]))
    
    # for i, x in enumerate(data[:,6]):
    #     if data[i,6] > 0:
    #       data[i,6] = 0

    data[:,-1] = np.exp(data[:,-1])

    model = regression_model(data)
    
    #model.restore_model()
    
    return model




