

import glob
import numpy as np
from efficientnet_pytorch import EfficientNet
import cv2  
import torch
import pickle

from keras import backend as K
K.clear_session()
import tensorflow as tf
#tf.reset_default_graph()
tf.compat.v1.reset_default_graph

#Xception
#ResNet50
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from moviepy.editor import VideoFileClip
import pandas as pd


import h5py


def getSecFromTimeStr(tm):
    tmTkn=tm.split(":")
    hr=int(tmTkn[0])
    mtn=int(tmTkn[1])
    sec=float(tmTkn[2])
    finSecs=(hr*3600)+(mtn*60)+(sec)
    return finSecs

def getTextBertPerSecSampleArr(sub,durationA,nSample,bert_embedding):
    caption=webvtt.read(sub)
    allTextPerSec=[]
    for i in range(durationA):
        allTextPerSec.append([""])
    for i in range(len(caption)):
        strt=int(getSecFromTimeStr(caption[i].start))
        ed=int(getSecFromTimeStr(caption[i].end))
        for x in range(strt,ed+1):
            allTextPerSec[x-1].append(caption[i].text)
    allTextSampleBert=[]
    for i in range(len(allTextPerSec)):
        bert_i = bert_embedding(allTextPerSec[i])
        for n in range(nSample):
            allTextSampleBert.append(bert_i)
    return np.array(allTextSampleBert)

def getVidFramesAndFpsDur(vidName,nSamples=2):
    video_clip = VideoFileClip(vidName)
    frmAll=list(video_clip.iter_frames(fps=nSamples,with_times=False))
    frmAll,fps,duration=frmAll,video_clip.fps,video_clip.duration
    del video_clip
    return frmAll,fps,duration

def getAudioFramesAndFpsDur(vidName):
    video_clip = VideoFileClip(vidName)
    audioclip = video_clip.audio
    frmAll=list(audioclip.iter_frames(with_times=False))
    frmAll,fps,duration=frmAll,audioclip.fps,audioclip.duration
    del video_clip
    return frmAll,fps,duration



def getGtForVideo(baseGtAnnotateData,vidName,users,nScorePerSeg,nFrames):
    vidGtDict={}
    vidGtDictAvg={}
    vidName=vidName+".mp4"
    for u in users:
        scores=baseGtAnnotateData[u][vidName]
        processedScore=[]
        for s in range(len(scores)):
            if(s<(len(scores)-1)):
                processedScore+=np.full(nScorePerSeg,scores[s]).tolist()
            else:
                processedScore+=np.full(nFrames%nScorePerSeg,scores[s]).tolist()
        if(vidName in vidGtDict):
            vidGtDict[vidName].append(processedScore)
        else:
            vidGtDict[vidName]=[processedScore]
    for k in vidGtDict:
        vidGtDictAvg[k]=np.array(vidGtDict[k]).mean(axis=0)
    return vidGtDictAvg[k]

def writeVisualFeaturesAlongGtTvSumForAllExtractorsMulti(basedir,justGt,baseOutGt,baseGtAnnotateData,baseOutEff,baseOutVgg,baseOutIncepV3,baseOutXception,baseOuResNet50,modelXception,modelResNet50,modelVgg,modelInceptionV3,videoNames,nSamples=2,strt=0):
    allVideos=glob.glob(basedir)
    for v in range(strt,len(allVideos)):
        vid=allVideos[v]
        vidName=vid.split("/")[-1][:-4]
        nameShowLimit=25
        if(not vidName in videoNames): #check for already done videos
            print("video num: ",v," name: ",vidName[:nameShowLimit]," not found labeled")
            continue
        sampledFrames,fps,duration=getVidFramesAndFpsDur(vid,nSamples)
        nFrames=len(sampledFrames)
        segSize=5
        nScorePerSeg=segSize*nSamples
        users=["junaid"]
        vidGt=getGtForVideo(baseGtAnnotateData,vidName,users,nScorePerSeg,nFrames)


        sampledFramesGt=vidGt
        fps=int(fps)

        if(not justGt):
            featuresArrMainVGG=[]
            featuresArrMainIncepV3=[]
            featuresArrMainXception=[]
            featuresArrMainResNet50=[]
#            featuresArrMainEff=[]
            for f in range(len(sampledFrames)):  
                print("video num: ",v," name: ",vidName[:nameShowLimit]," at frame: ",f)                     
                frmRz224=cv2.resize(sampledFrames[f], (224,224))
                frmRz299=cv2.resize(sampledFrames[f], (299,299))

                frmRz299=np.expand_dims(frmRz299, axis=0)
                frmRz224=np.expand_dims(frmRz224, axis=0)
                frmRz224Pi = preprocess_input(frmRz224)
                vgg16_feature = modelVgg.predict(frmRz224Pi)
                resnet50_feature = modelResNet50.predict(frmRz224)
                inceptionv3_feature = modelInceptionV3.predict(frmRz299)
                xception_feature = modelXception.predict(frmRz299)
                
                featuresArrVgg=vgg16_feature
                featuresArrIncepV3=inceptionv3_feature
                featuresArrXception=xception_feature
                featuresArrResNet50=resnet50_feature
                featuresArrMainVGG.append(featuresArrVgg)
                featuresArrMainIncepV3.append(featuresArrIncepV3)
                featuresArrMainXception.append(featuresArrXception)
                featuresArrMainResNet50.append(featuresArrResNet50)
            featuresArrMainVGG=np.array(featuresArrMainVGG)
            featuresArrMainIncepV3=np.array(featuresArrMainIncepV3)
            featuresArrMainXception=np.array(featuresArrMainXception)
            featuresArrMainResNet50=np.array(featuresArrMainResNet50)

            sampledFramesGt=np.array(sampledFramesGt)
 
            pickle.dump(featuresArrMainVGG,open(baseOutVgg[0]+vidName+"_"+baseOutVgg[1]+".p","wb"))
            pickle.dump(featuresArrMainIncepV3,open(baseOutIncepV3[0]+vidName+"_"+baseOutIncepV3[1]+".p","wb"))
            pickle.dump(featuresArrMainXception,open(baseOutXception[0]+vidName+"_"+baseOutXception[1]+".p","wb"))
            pickle.dump(featuresArrMainResNet50,open(baseOuResNet50[0]+vidName+"_"+baseOuResNet50[1]+".p","wb"))
            del featuresArrMainVGG
            del featuresArrMainIncepV3
            del featuresArrMainXception
            del featuresArrMainResNet50
            del sampledFrames
        pickle.dump(sampledFramesGt,open(baseOutGt+vidName+"_"+"smpGt"+".p","wb"))
        print("features and gt written for: ",vidName," vid:",v)
        print("frames and gt written for: ",vidName," vid:",v)
        



# thsi is all names train + test files as in data directory
videoNames=pd.read_csv('./annotatedData/annotatedVideos.csv',header=None,delimiter="XYZ").values 


differentBaseOut="./data/"
basedir="./AnnotatedVideosAndSubtitles_version_1/*.mp4"
baseOutEff=[differentBaseOut+"efficientNetB0Features/","efficientNetB0Features"]
baseOutVgg=[differentBaseOut+"vgg16Features/","vgg16Features"]
baseOutIncepV3=[differentBaseOut+"InceptionV3Features/","InceptionV3Features"]

baseOutXception=[differentBaseOut+"XceptionFeatures/","XceptionFeatures"]
baseOuResNet50=[differentBaseOut+"ResNet50Features/","ResNet50Features"]

baseOutGt=differentBaseOut+"gtAvg/"
baseGtAnnotate="./AnnotatedVideosAndSubtitles_version_1/annotations/annotations_version_1.p"

users=["user1"]
baseGtAnnotateData=pickle.load(open(baseGtAnnotate,"rb"))


modelVgg = VGG16(weights='imagenet', include_top=False,pooling='avg')

modelInceptionV3=InceptionV3(weights='imagenet', include_top=False,pooling='avg') # for v1 https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14

modelXception=Xception(weights='imagenet', include_top=False,pooling='avg')
modelResNet50=ResNet50(weights='imagenet', include_top=False,pooling='avg')



strt=1
justGt=True
nSamples=3
writeVisualFeaturesAlongGtTvSumForAllExtractorsMulti(basedir,justGt,baseOutGt,baseGtAnnotateData,baseOutEff,baseOutVgg,baseOutIncepV3,baseOutXception,baseOuResNet50,modelXception,modelResNet50,modelVgg,modelInceptionV3,videoNames,nSamples,strt=strt)

