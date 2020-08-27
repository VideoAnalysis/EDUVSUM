
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pickle
from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata
import pandas as pd
import time


#def getFileListWithGivenTag(dataBase,tag):

def getOneHotClasses(gt,numClasses):
    oneHot=[]
    for i in range(len(gt)):
        curr=np.zeros(numClasses).astype(int)
        curr[int(gt[i]-1)]=1
        oneHot.append(curr)
    return np.array(oneHot)

def getFtsFinal(fts,inShape,rsp,modelName):
    if(modelName=="ConvLSTM"):
        return fts
    else:
        fts1=fts.reshape((fts.shape[0],inShape[1],rsp))
        return np.mean(fts1,axis=2)
    
def getFileName(featureTag):
    if(featureTag=="299"):
        return "InceptionV3Featues","InceptionV3FeatuesFrames"
    elif(featureTag=="224"):
        return "vgg16Featues","vgg16FeatuesFrames"
    else:
        return featureTag,featureTag

def getTimeElapsed(strTime):
    totalTime=time.time()-strTime
    return [totalTime,totalTime/60,totalTime/3600]

def getFeatureShape(featureTag,hist):
    if(featureTag=="efficientNetB0Featues"):
        return 7*7,1280,(hist,1280)
    if(featureTag=="vgg16Featues"):
        return 7*7,512,(hist,512)
    if(featureTag=="inceptionV3Featues"):
        return 5*5,2048,(hist,2048)
    if(featureTag=="299"):
        return 7*7,299,(hist,299,299,3)
    if(featureTag=="224"):
        return 7*7,299,(hist,224,224,3)
    
def getReshape(fts,inShape,modelName,batch):
    if(modelName=="LSTM" or modelName=="BiLSTM"):
        return fts.reshape((batch,inShape[0],inShape[1]))
    if(modelName=="MLP"):
        return fts.reshape((batch,inShape[0]*inShape[1]))
    if(modelName=="ConvLSTM"):
        return fts.reshape((batch,inShape[0],inShape[1],inShape[2],inShape[3]))
        
def getVideoList(videoList):
    dfArr=pd.read_csv(videoList,header=None,sep="JUNAID").values.tolist()
    allFilesTvSum=[]
    for v in dfArr:
        allFilesTvSum.append(v[0])
    return allFilesTvSum

def getLossAndResults(model,allFilesTvSumTest,hist,fbase,gtbase,inShape,rsp,tvSumBase,modelName):
    testLossLs=[]
    yNyhatByVideo=[]
    testMaeLossLs=[]
    for j in range(len(allFilesTvSumTest)):
        Y=[]
        yhatLs=[]
        fts=pickle.load(open(tvSumBase+allFilesTvSumTest[j]+fbase,"rb"))
        fts1=fts.reshape((fts.shape[0],inShape[1],rsp))
        fts2=np.mean(fts1,axis=2)
        gt=pickle.load(open(tvSumBase+allFilesTvSumTest[j]+gtbase,"rb"))
        for h in range(hist,len(gt)):
            xtest=getReshape(fts2[h-hist:h],inShape,modelName,1)
            yhat=model.predict(xtest)
            yhatLs.append(yhat[0][0])
            Y.append(gt[h-1])
        tMse=(np.square(np.array(Y) - np.array(yhatLs))).mean(axis=0)
        pS=spearmanr(np.array(Y),np.array(yhatLs))[0]
        kT=kendalltau(rankdata(-np.array(Y)), rankdata(-np.array(yhatLs)))[0]
        tMae=(np.abs(np.array(Y) - np.array(yhatLs))).mean(axis=0)
        yNyhatByVideo.append([allFilesTvSumTest[j],Y,yhatLs])
        testLossLs.append([tMse,pS,kT])
        testMaeLossLs.append(tMae)
        testLossMean=np.array(testLossLs).mean(axis=0)
    return testLossMean,testLossLs,yNyhatByVideo,testMaeLossLs


def getMseLossConvLstm(model,allFilesTvSumTest,hist,fbase,gtbase,inShape,rsp,tvSumBase):
    testLossLs=[]
    testLossLsMae=[]
    yNyhatByVideo=[]
    for j in range(len(allFilesTvSumTest)):
        Y=[]
        yhatLs=[]
        fts=pickle.load(open(tvSumBase+allFilesTvSumTest[j]+fbase,"rb"))
        gt=pickle.load(open(tvSumBase+allFilesTvSumTest[j]+gtbase,"rb"))
        for h in range(hist,len(gt)):
            xtest=fts[h-hist:h].reshape((1,inShape[0],inShape[1],inShape[2],inShape[3]))
            yhat=model.predict(xtest)
            yhatLs.append(yhat[0][0])
            Y.append(gt[h-1])
        yNyhatByVideo.append([allFilesTvSumTest[j],Y,yhatLs])
        tMse=(np.square(np.array(Y) - np.array(yhatLs))).mean(axis=0)
        tMae=(np.abs(np.array(Y) - np.array(yhatLs))).mean(axis=0)
        testLossLsMae.append(tMae)
        testLossLs.append(tMse)
    return np.mean(testLossLs),testLossLs,yNyhatByVideo,testLossLsMae