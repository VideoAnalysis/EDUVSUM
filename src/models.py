#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM,Bidirectional
from keras.datasets import imdb
import glob
import numpy as np
import time
import pickle
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization



def getModel(params,modelName,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth,classes):
#    if(modelName == "triFusion_v_a_t"):
    if("triFusion" in modelName):
        return getLstmMultiModelTriFusion2(params,modelName,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth,classes)
    elif("triRegFusion" in modelName):
        return getLstmMultiModelTriFusion2reg(params,modelName,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth,classes)
#        return getLstmMultiModelTriFusion_v_a_t(params,modelName,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth,classes)
    if(modelName == "biTriFusion"):
        return getLstmMultiModelBiTriFusion(params,modelName,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth,classes)


def getLstmMultiModelTriFusion_v_a_t(params,modelName,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth,classes):
    print('Build model...')
    lstmLayers=[64,64]
    denseLayers=[32,16,classes]
    inputV = keras.layers.Input(shape=(inputShapeV[0],inputShapeV[1]))
    inputA = keras.layers.Input(shape=(inputShapeA[0],inputShapeA[1]))
    inputT = keras.layers.Input(shape=(inputShapeT[0],inputShapeT[1]))
    lstmT=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputT)#,return_sequences=True
    lstmA=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputA)
    lstmV=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputV)
    lstOutTAV=keras.layers.concatenate([lstmT,lstmA,lstmV], axis=1)
    lstmTAV=keras.layers.LSTM(lstmLayers[0], dropout=0.2, recurrent_dropout=0.2)(lstOutTAV) # another variation to apply bidirectional after fusion as well
    d1=keras.layers.Dense(denseLayers[0], activation=acth)(lstmTAV)
    d2=keras.layers.Dense(denseLayers[1], activation=acth)(d1)
#    out = keras.layers.Dense(denseLayers[2], activation=actf)(d2)
    out = keras.layers.Dense(denseLayers[2], activation=actf)(d2)
    model = keras.models.Model(inputs=[inputV,inputA,inputT], outputs=out)
#    model.compile(loss='mean_squared_error',optimizer=opt,metrics=['mse'])
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy']) #categorical_crossentropy #mean_squared_error #accuracy #mse
    return model

def getLstmMultiModelTriFusion2(params,modelName,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth,classes):
    print('Build model...')
    visualIndicator=params["visualIndicator"]
    audioIndicator=params["audioIndicator"]
    textIndicator=params["textIndicator"]
    lstmLayers=[64,64]
    denseLayers=[32,16,classes]
    inputV = keras.layers.Input(shape=(inputShapeV[0],inputShapeV[1]))
    inputA = keras.layers.Input(shape=(inputShapeA[0],inputShapeA[1]))
    inputT = keras.layers.Input(shape=(inputShapeT[0],inputShapeT[1]))
    lstmT=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputT)#,return_sequences=True
    lstmA=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputA)
    lstmV=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputV)
    lstOutTAV=keras.layers.concatenate([lstmV,lstmA,lstmT], axis=1)
    if(not visualIndicator in modelName):
        lstOutTAV=keras.layers.concatenate([lstmA,lstmT], axis=1)  
    elif(not audioIndicator in modelName):
        lstOutTAV=keras.layers.concatenate([lstmV,lstmT], axis=1)
    elif(not textIndicator in modelName):
        lstOutTAV=keras.layers.concatenate([lstmV,lstmA], axis=1)    
#    else:
        
    lstmTAV=Bidirectional(keras.layers.LSTM(lstmLayers[0], dropout=0.2, recurrent_dropout=0.2))(lstOutTAV)
    d1=keras.layers.Dense(denseLayers[0], activation=acth)(lstmTAV)
    d2=keras.layers.Dense(denseLayers[1], activation=acth)(d1)
#    out = keras.layers.Dense(denseLayers[2], activation=actf)(d2)
    out = keras.layers.Dense(denseLayers[2], activation=actf)(d2)
    model = keras.models.Model(inputs=[inputV,inputA,inputT], outputs=out)
    if(not visualIndicator in modelName):
        model = keras.models.Model(inputs=[inputA,inputT], outputs=out)
    elif(not audioIndicator in modelName):
        model = keras.models.Model(inputs=[inputV,inputT], outputs=out)
    elif(not textIndicator in modelName):
        model = keras.models.Model(inputs=[inputV,inputA], outputs=out)    
#    model.compile(loss='mean_squared_error',optimizer=opt,metrics=['mse'])
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy']) #categorical_crossentropy #mean_squared_error #accuracy #mse
    return model

def getLstmMultiModelTriFusion2reg(params,modelName,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth,classes):
    print('Build model...')
    visualIndicator=params["visualIndicator"]
    audioIndicator=params["audioIndicator"]
    textIndicator=params["textIndicator"]
    lstmLayers=[64,64]
    denseLayers=[32,16,1] # here no classes concppt
    inputV = keras.layers.Input(shape=(inputShapeV[0],inputShapeV[1]))
    inputA = keras.layers.Input(shape=(inputShapeA[0],inputShapeA[1]))
    inputT = keras.layers.Input(shape=(inputShapeT[0],inputShapeT[1]))
    lstmT=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputT)#,return_sequences=True
    lstmA=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputA)
    lstmV=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputV)
    lstOutTAV=keras.layers.concatenate([lstmV,lstmA,lstmT], axis=1)
    if(not visualIndicator in modelName):
        lstOutTAV=keras.layers.concatenate([lstmA,lstmT], axis=1)  
    elif(not audioIndicator in modelName):
        lstOutTAV=keras.layers.concatenate([lstmV,lstmT], axis=1)
    elif(not textIndicator in modelName):
        lstOutTAV=keras.layers.concatenate([lstmV,lstmA], axis=1)    
#    else:
        
    lstmTAV=keras.layers.LSTM(lstmLayers[0], dropout=0.2, recurrent_dropout=0.2)(lstOutTAV)
    d1=keras.layers.Dense(denseLayers[0], activation=acth)(lstmTAV)
    d2=keras.layers.Dense(denseLayers[1], activation=acth)(d1)
#    out = keras.layers.Dense(denseLayers[2], activation=actf)(d2)
    out = keras.layers.Dense(denseLayers[2], activation=actf)(d2)
    model = keras.models.Model(inputs=[inputV,inputA,inputT], outputs=out)
    if(not visualIndicator in modelName):
        model = keras.models.Model(inputs=[inputA,inputT], outputs=out)
    elif(not audioIndicator in modelName):
        model = keras.models.Model(inputs=[inputV,inputT], outputs=out)
    elif(not textIndicator in modelName):
        model = keras.models.Model(inputs=[inputV,inputA], outputs=out)    
#    model.compile(loss='mean_squared_error',optimizer=opt,metrics=['mse'])
    model.compile(loss='mean_squared_error',optimizer=opt,metrics=['accuracy']) #categorical_crossentropy #mean_squared_error #accuracy #mse
    return model


def getLstmMultiModelTriFusion(params,modelName,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth,classes):
    print('Build model...')
    visualIndicator=params["visualIndicator"]
    audioIndicator=params["audioIndicator"]
    textIndicator=params["textIndicator"]
    lstmLayers=[64,64]
    denseLayers=[32,16,classes]
    inputV=0
    inputA=0
    inputT=0
    lstmV=0
    lstmA=0
    lstmT=0
    if(visualIndicator in modelName):
        inputV = keras.layers.Input(shape=(inputShapeV[0],inputShapeV[1]))
        lstmV=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputV)
    if(audioIndicator in modelName):
        inputA = keras.layers.Input(shape=(inputShapeA[0],inputShapeA[1]))
        lstmA=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputA)
    if(textIndicator in modelName):
        inputT = keras.layers.Input(shape=(inputShapeT[0],inputShapeT[1]))
        lstmT=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputT)#,return_sequences=True
    
    inputToConcat=[x for x in [lstmT,lstmA,lstmV] if x!=0]#[lstmT,lstmA,lstmV]
    concatInputs=keras.layers.concatenate(inputToConcat, axis=1)
    lstmInputs=keras.layers.LSTM(lstmLayers[0], dropout=0.2, recurrent_dropout=0.2)(concatInputs)
    d1=keras.layers.Dense(denseLayers[0], activation=acth)(lstmInputs)
    d2=keras.layers.Dense(denseLayers[1], activation=acth)(d1)
#    out = keras.layers.Dense(denseLayers[2], activation=actf)(d2)
    out = keras.layers.Dense(denseLayers[2], activation=actf)(d2)
    inputToModel=[x for x in [inputV,inputA,inputT] if x!=0]#[inputV,inputA,inputT]
    model = keras.models.Model(inputs=inputToModel, outputs=out)
#    model.compile(loss='mean_squared_error',optimizer=opt,metrics=['mse'])
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy']) #categorical_crossentropy #mean_squared_error #accuracy #mse
    return model

#def getLstmMultiModelTriFusion_v_a_t(indicators,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth,classes=10):
#    print('Build model...')
#    lstmLayers=[64,64]
#    denseLayers=[32,16,classes]
#    inputV = keras.layers.Input(shape=(inputShapeV[0],inputShapeV[1]))
#    inputA = keras.layers.Input(shape=(inputShapeA[0],inputShapeA[1]))
#    inputT = keras.layers.Input(shape=(inputShapeT[0],inputShapeT[1]))
#    lstmT=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputT)#,return_sequences=True
#    lstmA=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputA)
#    lstmV=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputV)
#    lstOutTAV=keras.layers.concatenate([lstmT,lstmA,lstmV], axis=1)
#    lstmTAV=keras.layers.LSTM(lstmLayers[0], dropout=0.2, recurrent_dropout=0.2)(lstOutTAV)
#    d1=keras.layers.Dense(denseLayers[0], activation=acth)(lstmTAV)
#    d2=keras.layers.Dense(denseLayers[1], activation=acth)(d1)
##    out = keras.layers.Dense(denseLayers[2], activation=actf)(d2)
#    out = keras.layers.Dense(denseLayers[2], activation=actf)(d2)
#    model = keras.models.Model(inputs=[inputV,inputA,inputT], outputs=out)
##    model.compile(loss='mean_squared_error',optimizer=opt,metrics=['mse'])
#    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy']) #categorical_crossentropy #mean_squared_error #accuracy #mse
#    return model



def getLstmMultiModelBiTriFusion(params,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth):
    print('Build model...')
    lstmLayers=[64,64]
    denseLayers=[32,16,1]
    inputV = keras.layers.Input(shape=(inputShapeV[0],inputShapeV[1]))
    inputA = keras.layers.Input(shape=(inputShapeA[0],inputShapeA[1]))
    inputT = keras.layers.Input(shape=(inputShapeT[0],inputShapeT[1]))
    lstmT=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputT)#,return_sequences=True
    lstmA=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputA)
    lstmV=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputV)
    lstmOutTA=keras.layers.concatenate([lstmT,lstmA], axis=-1)
    lstmOutTV=keras.layers.concatenate([lstmT,lstmV], axis=-1)
    lstmOutAV=keras.layers.concatenate([lstmA,lstmV], axis=-1)
    lstmTA=Bidirectional(keras.layers.LSTM(lstmLayers[0], dropout=0.2, recurrent_dropout=0.2))(lstmOutTA)
    lstmTV=Bidirectional(keras.layers.LSTM(lstmLayers[0], dropout=0.2, recurrent_dropout=0.2))(lstmOutTV)
    lstmAV=Bidirectional(keras.layers.LSTM(lstmLayers[0], dropout=0.2, recurrent_dropout=0.2))(lstmOutAV)
    lstmOutTAV=keras.layers.concatenate([lstmTA,lstmTV,lstmAV], axis=-1)
    d1=keras.layers.Dense(denseLayers[0], activation=acth)(lstmOutTAV)
    d2=keras.layers.Dense(denseLayers[1], activation=acth)(d1)
    out = keras.layers.Dense(denseLayers[2], activation=actf)(d2)
    model = keras.models.Model(inputs=[inputV,inputA,inputT], outputs=out)
    model.compile(loss='mean_squared_error',optimizer=opt,metrics=['mae'])
    return model


def getLSTMmodel(inputShape,optimizerName,activationName):
    print('Build model...')
    lstmLayers=[64,64]
    denseLayers=[32,16,1]
    input1 = keras.layers.Input(shape=(inputShape[0],inputShape[1]))
    ls1=keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input1)
    ls2=keras.layers.LSTM(lstmLayers[0], dropout=0.2, recurrent_dropout=0.2)(ls1)
    d1=keras.layers.Dense(denseLayers[0], activation=activationName)(ls2)
    d2=keras.layers.Dense(denseLayers[1], activation=activationName)(d1)
    out = keras.layers.Dense(denseLayers[2], activation=activationName)(d2)
    model = keras.models.Model(inputs=[input1], outputs=out)
    model.compile(loss='mean_squared_error',optimizer=optimizerName,metrics=['mae'])
    return model


def getBiLSTMmodel(inputShape,optimizerName,activationName):
    print('Build model...')
    lstmLayers=[64,64]
    denseLayers=[32,16,1]
    input1 = keras.layers.Input(shape=(inputShape[0],inputShape[1]))
    ls1=Bidirectional(keras.layers.LSTM(lstmLayers[0],return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(input1)
    ls2=Bidirectional(keras.layers.LSTM(lstmLayers[0], dropout=0.2, recurrent_dropout=0.2))(ls1)
    d1=keras.layers.Dense(denseLayers[0], activation=activationName)(ls2)
    d2=keras.layers.Dense(denseLayers[1], activation=activationName)(d1)
    out = keras.layers.Dense(denseLayers[2])(d2)
    model = keras.models.Model(inputs=[input1], outputs=out)
    model.compile(loss='mean_squared_error',optimizer=optimizerName,metrics=['mae'])
    return model

def getMLPmodel(inShape,opt,actf):
    print('Build model...')
    denseLayers=[64,64,32,16,1]
    model = Sequential([
          Dense(denseLayers[0], activation=actf, input_shape=(inShape[0]*inShape[1],)),
          Dense(denseLayers[1], activation=actf),
          Dense(denseLayers[2], activation=actf),
          Dense(denseLayers[3], activation=actf),
          Dense(denseLayers[4], activation=actf),
        ])
    
    model.compile(
      optimizer=opt,
      loss='mean_squared_error',
      metrics=['mse'],
    )
    return model

def getLstmModelConvLstmWoPool(inputShape,optimizerName,activationName):
    print('Build model...')
    filters=[40,40]
    denseLayers=[32,16,1]
    input1 = keras.layers.Input(shape=inputShape)
    ls1=ConvLSTM2D(filters=filters[0], kernel_size=(3, 3),
                   input_shape=inputShape, #(None, 40, 40, 1)
                   padding='same', return_sequences=True)(input1)
    bn1=BatchNormalization()(ls1)

    ls2=ConvLSTM2D(filters=filters[1], kernel_size=(3, 3),
                   padding='same')(bn1)
    bn2=BatchNormalization()(ls2)

    flat_layer = keras.layers.Flatten()(bn2)
    d1=keras.layers.Dense(denseLayers[0], activation=activationName)(flat_layer)
    d2=keras.layers.Dense(denseLayers[1], activation=activationName)(d1)
    out = keras.layers.Dense(denseLayers[2])(d2)
    model = keras.models.Model(inputs=[input1], outputs=out)
    model.compile(loss='mean_squared_error',optimizer=optimizerName,metrics=['mse'])
    return model


