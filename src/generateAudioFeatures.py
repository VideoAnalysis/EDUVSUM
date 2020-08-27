

from common import videoLoading
import glob
import numpy as np
import time
import os
import subprocess
import pickle
#https://pypi.org/project/webvtt-py/
#https://github.com/tyiannak/pyAudioAnalysis/



def getSecFromTimeStr(tm):
    tmTkn=tm.split(":")
    hr=int(tmTkn[0])
    mtn=int(tmTkn[1])
    sec=float(tmTkn[2])
    finSecs=(hr*3600)+(mtn*60)+(sec)
    return finSecs

def getTextPerSecsArr(vidName,durationA):
    caption=webvtt.read(vidName)
    allTextPerSec=[]
    for i in range(durationA):
        allTextPerSec.append([""])
    for i in range(len(caption)):
        strt=int(getSecFromTimeStr(caption[i].start))
        ed=int(getSecFromTimeStr(caption[i].end))
        for x in range(strt,ed+1):
            allTextPerSec[x-1].append(caption[i].text)
    return np.array(allTextPerSec)

def getTextPerSecsArrByChunks(vidName,durationA,chunk):
    caption=webvtt.read(vidName)
    allTextPerSecByChunk=[]
    for i in range(durationA//chunk):
        allTextPerSecByChunk.append([""])
    for i in range(len(caption)):
        strt=int(getSecFromTimeStr(caption[i].start))
        ed=int(getSecFromTimeStr(caption[i].end))
        allTextPerSecByChunk[strt//chunk].append(caption[i].text)
        allTextPerSecByChunk[ed//chunk].append(caption[i].text)
    return np.array(allTextPerSecByChunk)
#for caption in webvtt.read(allVideosSubs[0]):
#    print(caption.start)
#    print(caption.end)
#    print(caption.text)
    
  
def getAvgGt(gtDf,vidName):
    subGt=gtDf.loc[vidName]
    allUserArr=[]
    for i in range(len(subGt)):
        allUserArr.append(subGt.iloc[i,-1].split(","))
    allUserArr=np.array(allUserArr)
    allUserArr=allUserArr.astype(int)
    meanGt=allUserArr.mean(axis=0)
    return meanGt

def getAudioDataFromVideo(vid,audioSamplingRate):
    frmAll,fps,durationA=videoLoading.getAudioFramesAndFpsDur(vid)
    fps=int(fps)
    sampledFramesA=[]
    nSamples=int(fps*audioSamplingRate)
    for i in range(int(durationA)):
        smpl=np.sort(np.random.choice(np.arange(fps),nSamples,replace=False))
        for idx in smpl:
            sampledFramesA.append(frmAll[i*fps:(i+1)*fps][idx])
#        print("on duration: ",i," sith sample: ",nSamples)
    return sampledFramesA,fps,durationA,nSamples

#from pandas import DataFrame
#gtDf = DataFrame.from_csv(groundTruth, sep="\t",header=None)
chunk=5
audioSamplingRate=0.10

def getAudioDataAll(basedir,baseOut,chunk,audioSamplingRate):
    allVideos=glob.glob(basedir)
    for v in range(len(allVideos)):
        vid=allVideos[v]
        vidName=vid.split("/")[-1].split(".")[0]
        frmAll,fps,durationA=videoLoading.getAudioFramesAndFpsDur(vid)
        fps=int(fps)
        sampledFramesA=[]
        nSamples=int(fps*audioSamplingRate)
        for i in range(int(durationA)):
            smpl=np.sort(np.random.choice(np.arange(fps),nSamples,replace=False))
            for idx in smpl:
                sampledFramesA.append(frmAll[i*fps:(i+1)*fps][idx])
            print("on duration: ",i," sith sample: ",nSamples)
        sampledFramesA=np.array(sampledFramesA)
        pickle.dump(sampledFramesA,open(baseOut+vidName+"_"+str(nSamples)+"_sampledFramesAudio.p","wb"))
        print("audio and text written for: ",vidName)



def getAudioDataWithFeaturesAll(basedir,baseOut,audioSamplingRate,annotatedVideos,doneAudioList):
    allVideos=glob.glob(basedir+"*.mp4")
    for v in range(len(allVideos)):
        vid=allVideos[v]
        vidName=vid.split("/")[-1][:-4]
        nameShowLimit=25
        if((not vidName in annotatedVideos) or vidName in doneAudioList): #check for already done videos
            print("video num: ",v," name: ",vidName[:nameShowLimit]," not found labeled")
            continue
        os.system("nohup ffmpeg -i '"+vid+"' "+baseOut+str(v)+".wav &")
        time.sleep(5)
        print("wait after audio.wav ..")
        os.system("nohup ffmpeg -i "+baseOut+str(v)+".wav -ar 24000 -ac 1 '"+baseOut+vidName+"_24K_mono.wav' &")
        time.sleep(5)
        print("wait after 24k_mono.wav ..")
        frmAll,fps,durationA=videoLoading.getAudioFramesAndFpsDur(vid)
        print("read video and audio processing..")
        fps=int(fps)
        sampledFramesA=[]
        nSamples=int(fps*audioSamplingRate)
        for i in range(int(durationA)):
            smpl=np.sort(np.random.choice(np.arange(fps),nSamples,replace=False))
            for idx in smpl:
                sampledFramesA.append(frmAll[i*fps:(i+1)*fps][idx])
        sampledFramesA=np.array(sampledFramesA)
        audio_data=baseOut+vidName+"_24K_mono.wav"
        [Fs, x] = audioBasicIO.read_audio_file(audio_data)
        windowSz=0.050
        stepSz=0.025
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, windowSz*Fs, stepSz*Fs)
        pickle.dump([F, f_names],open(baseOut+vidName+"_"+str(windowSz)+"_"+str(stepSz)+"_featuresAudio.p","wb"))
        pickle.dump(sampledFramesA,open(baseOut+vidName+"_"+str(nSamples)+"_sampledFramesAudio.p","wb"))
        print("video num: ",v," audio written for: ",vidName[:nameShowLimit])




def energy_entropy(frame, n_short_blocks=20):
    """Computes entropy of energy"""
    # total frame energy
    eps = 0.00000001
    frame_energy = np.sum(frame ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy



from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import matplotlib.pyplot as plt
import pandas as pd


# thsi is all names train + test files as in data directory
annotatedVideos=pd.read_csv('./annotatedData/annotatedVideos.csv',header=None,delimiter="XYZ").values 

doneAudioBase="./data/audioFeatures/*.wav"
doneAudio=glob.glob(doneAudioBase)
doneAudioList=[x.split("/")[-1][:-13] for x in doneAudio ]

basedirVids="./AnnotatedVideosAndSubtitles_version_1/"
baseOut="./data/audioFeatures/"

allVideos=glob.glob(basedirVids+"*.mp4")

audioSamplingRate=0.10

getAudioDataWithFeaturesAll(basedirVids,baseOut,audioSamplingRate,annotatedVideos,doneAudioList)


