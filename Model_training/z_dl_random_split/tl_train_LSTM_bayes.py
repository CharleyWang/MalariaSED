from genome_2_tensor import tensorLoad
from operator import itemgetter
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from random import sample,seed
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import max_norm
from keras_tuner import RandomSearch
from keras_tuner import Hyperband, BayesianOptimization
import keras_tuner as kt
import json
import os
import math

def train_testID_returnSampling(inputID_file, tfX, ratioT = 0.15):
    with open (inputID_file, 'r') as f:
        lines = f.readlines()
    seed(0)
    ids = sample(range(len(lines)), len(lines))
    numTest = math.floor(ratioT * len(lines))
    numValid= math.floor(ratioT * len(lines))
    idValid = ids[:numValid]
    idTest  = ids[numValid:(numValid + numTest)]
    idTrain= ids[(numValid + numTest):]
    
    print(len(idValid), len(idTest), len(idTrain))
    tfXtest   = tf.gather(tfX, indices = idTest)
    tfXvalid  = tf.gather(tfX, indices = idValid)
    tfXtrain  = tf.gather(tfX, indices = idTrain)
    return((tfXtrain, tfXvalid, tfXtest))
    

#from tensor y sample a propotion relative to x alone axis = 0
def sampleSamePropotion(x, y, propotion = 1):
    id = sample(range(y.shape[0]), x.shape[0] * propotion)
    return(tf.gather(y, indices = id))

#shuffle input x y
def shuffleData(x, y):
    id = sample(range(y.shape[0]),y.shape[0])
    return(tf.gather(x, indices = id),
           tf.gather(y, indices = id))

def make_model( hp ):
    METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'), 
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    i = 1
    model = keras.models.Sequential([
        keras.layers.Conv1D(
            hp.Int('convKnernal1st',240,320,80),
            hp.Int('kmer' + str(i), 6,10,1), 
            activation = hp.Choice('ConvAct_' + str(i), ['relu', 'tanh']),
            padding = hp.Choice('ConvPadding' + str(i), ['valid','same']),
            input_shape = [1000,4],
            kernel_regularizer=regularizers.l2(hp.Float('convL2' + str(i), 1e-10, 1e-4,sampling='log')),
            kernel_constraint=max_norm(hp.Float('max_norm_1stConv' + str(i), 0.1,3.1,0.5))
        ),
        keras.layers.MaxPooling1D(
            hp.Int('maxPool' + str(i), 2,6,1), 
            strides = hp.Int('maxPool_strid' + str(i), 2,6,1)
        ),
        keras.layers.Dropout(hp.Float('drop_' + str(i), 0.1, 0.9, 0.1)),
    ##cov1D 2ed
        keras.layers.Conv1D(
            480,
            hp.Int('kmer' + str(i + 1), 6,10,1), 
            activation = hp.Choice('ConvAct_' + str(i + 1), ['relu', 'tanh']),
            padding = hp.Choice('ConvPadding' + str(i + 1), ['valid','same']),
            kernel_regularizer=regularizers.l2(hp.Float('convL2' + str(i + 1), 1e-10, 1e-4,sampling='log')),
            kernel_constraint=max_norm(hp.Float('max_norm_1stConv' + str(i + 1), 0.1,3.1,0.5))
        ),
        keras.layers.MaxPooling1D(
            hp.Int('maxPool' + str(i + 1), 2,6,1), 
            strides = hp.Int('maxPool_strid' + str(i + 1), 2,6,1)
        ),
        keras.layers.Dropout(hp.Float('drop_' + str(i + 1), 0.1, 0.9, 0.1)),
        keras.layers.Bidirectional(keras.layers.LSTM(hp.Int('LSTM',30,320,60),
            kernel_regularizer=regularizers.l2(hp.Float('LSTML2', 1e-10, 1e-4,sampling='log')),
            kernel_constraint=max_norm(hp.Float('max_norm_LSTM', 0.1,3.1,0.5)),
            return_sequences=hp.Boolean('LSTM_sequence'))),
    
    ####################################################################
        keras.layers.Flatten(),
        keras.layers.Dense(
            hp.Int('densLayerKernal',40,160,40),
            activation = 'relu',
            kernel_regularizer=regularizers.l2(hp.Float('densL2', 1e-10, 1e-4,sampling='log')),
            kernel_constraint=max_norm(hp.Float('max_norm_Dens', 0.1,3.1,0.5)),
            activity_regularizer=regularizers.l2(hp.Float('densL2activate', 1e-10, 1e-4, sampling = 'log'))),
        keras.layers.Dense(1, activation = 'sigmoid',bias_initializer=None),
    ])
    model.compile(
            optimizer=keras.optimizers.Adam(lr=hp.Choice('ADAM',[1e-3,])),
            #optimizer=keras.optimizers.Adam(lr=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=METRICS)
    return model

def main():
    trainAllpos = tensorLoad('../time_specific_peak_generateTensor/tensorData_h35_40/positive.tensor_1k')
    trainAllneg = tensorLoad('../time_specific_peak_generateTensor/tensorData_h35_40/negative.tensor_1k')
    
    (tfXtrainPos, tfXvalidPos, tfXtestPos) = train_testID_returnSampling('../time_specific_peak_generateTensor/tensorData_h35_40/locName.positive_1k.txt', trainAllpos) #test valid

    (tfXtrainNeg, tfXvalidNeg, tfXtestNeg) = train_testID_returnSampling('../time_specific_peak_generateTensor/tensorData_h35_40/locName.negative_1k.txt', trainAllneg)
    print(tfXtrainPos.shape)
    print(tfXvalidPos.shape)
    print(tfXtestPos.shape)
    print('------pos------')
    print(tfXtrainNeg.shape)
    print(tfXvalidNeg.shape)
    print(tfXtestNeg.shape)
    print('------neg------')
    
    tfXtrainNegSample  = tfXtrainNeg#sampleSamePropotion(tfXtrainPos, tfXtrainNeg,10)
    tfXtestNegSample   = tfXtestNeg#sampleSamePropotion(tfXtestPos, tfXtestNeg,10)
    tfXvalidatNegSample= tfXvalidNeg

    trainY = tf.concat( [tf.repeat(1, tfXtrainPos.shape[0]),
            tf.repeat(0, tfXtrainNegSample.shape[0])], 0)
    testY  = tf.concat( [tf.repeat(1, tfXtestPos.shape[0]),
            tf.repeat(0, tfXtestNegSample.shape[0])], 0)
    valiY  = tf.concat( [tf.repeat(1, tfXvalidPos.shape[0]),
            tf.repeat(0, tfXvalidatNegSample.shape[0])], 0)
    
    trainX = tf.concat([tfXtrainPos, tfXtrainNegSample], 0)
    testX  = tf.concat([tfXtestPos , tfXtestNegSample] , 0)
    valiX  = tf.concat([tfXvalidPos , tfXvalidatNegSample],0)

    ##############shuffle the trainning data
    (trainX, trainY) = shuffleData(trainX, trainY)

    ##early_stoping
    early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_prc',
            verbose = 1,
            patience = 10,
            mode = 'max',
            restore_best_weights = True)

    mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', 
            monitor='val_prc', 
            mode='max', 
            verbose=1, 
            save_best_only=True)

    EPOCHS = 3000
    BATCH_size = 2000
    
    ##for multiple GPU
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))    
   
    tuner=BayesianOptimization(
            make_model,
            objective= kt.Objective('val_prc', direction = 'max'),
            max_trials= 5,
            )
    print( tuner.search_space_summary() )
    
    tuner.search(x = trainX,
            y = trainY,
            epochs = 3000,
            batch_size = BATCH_size,
            callbacks=[early_stopping],
            validation_data = (valiX, valiY))
    print(tuner.results_summary())

    models = tuner.get_best_models(num_models=2)
    model = models[0]
    model2 = models[1]
    model.summary()
    
    model.save('best_model1')
    model2.save('best_model2')
    modelLoad = tf.keras.models.load_model('best_model1')
    results1 = modelLoad.evaluate(valiX,valiY)
    results2 = modelLoad.evaluate(testX,testY)
    print(results1)
    print(results2)
    modelLoad = tf.keras.models.load_model('best_model2')
    results3 = modelLoad.evaluate(valiX,valiY)
    results4 = modelLoad.evaluate(testX,testY)
    print(results3)
    print(results4)



if __name__ == '__main__':
    tf.keras.backend.clear_session()
    main()
    #aa = make_model()
    #aa.summary()
#print(tfXtrainNeg.shape)
#print(tfXtrainNeg.shape[0])



#posID = pd.read_csv('../time_specific_peak_generateTensor/tensorData/locName.positive.txt', header = None)
#negID = pd.read_csv('../time_specific_peak_generateTensor/tensorData/locName.negative.txt', header = None)


#print(posID.head())
#print(pd.Series(range())posID[0].str.contains("chr13:").index)
