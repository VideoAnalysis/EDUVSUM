

from common import videoLoading
import glob
import numpy as np 
import pickle
import webvtt

#https://pypi.org/project/webvtt-py/



from bert_embedding import BertEmbedding
from transformers import pipeline
#https://pypi.org/project/bert-embedding/




def getSecFromTimeStr(tm):
    tmTkn=tm.split(":")
    hr=int(tmTkn[0])
    mtn=int(tmTkn[1])
    sec=float(tmTkn[2])
    finSecs=(hr*3600)+(mtn*60)+(sec)
    return finSecs

    


def getTextPerSecsArr(sub,durationA):
    caption=webvtt.read(sub)
    allTextPerSec=[]
    for i in range(int(durationA)):
        allTextPerSec.append([""])
    for i in range(len(caption)):
        strt=int(getSecFromTimeStr(caption[i].start))
        ed=int(getSecFromTimeStr(caption[i].end))
        for x in range(strt,ed+1):
            allTextPerSec[x-1].append(caption[i].text)
    return np.array(allTextPerSec)

def getAllTextInSubs(sub):
    caption=webvtt.read(sub)
    listofTexts=[]
    for i in range(len(caption)):
        listofTexts.append(caption[i].text)
    return listofTexts
         
def getTextPerSecsArrByChunks(sub,durationA,chunk):
    caption=webvtt.read(sub)
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

def getTextDataFromVideoPerSec(baseDir,vid,subs):
    frmAll,fps,durationA=videoLoading.getAudioFramesAndFpsDur(vid)
    allTextPerSec=getTextPerSecsArr(baseDir+subs,durationA)
    return allTextPerSec,fps,durationA

def getTextDataFromVideoPerChunk(vid,sub,chunk=5):
    frmAll,fps,durationA=videoLoading.getAudioFramesAndFpsDur(vid)
    allTextPerSecByChunk=getTextPerSecsArrByChunks(sub,durationA,chunk)
    return allTextPerSecByChunk,fps,durationA



def getTxtDataAll(basedir,baseOut,chunk):
    allVideos=glob.glob(basedir)
    for v in range(len(allVideos)):
        vid=allVideos[v]
        vidName=vid.split("/")[-1].split(".")[0]
        frmAll,fps,durationA=videoLoading.getAudioFramesAndFpsDur(vid)
        allTextPerSec=getTextPerSecsArr(vid,durationA)
        allTextPerSecByChunk=getTextPerSecsArrByChunks(vid,durationA,chunk)
        pickle.dump(allTextPerSec,open(baseOut+vidName+"_persec_allTextPerSec.p","wb"))
        pickle.dump(allTextPerSecByChunk,open(baseOut+vidName+"_"+str(chunk)+"_persec_allTextPerSecByChunk.p","wb"))
        print("audio and text written for: ",vidName)



def generateTextFeaturesForAllSec2(bert_embedding,bert_embedding2nlp,allTextPerSec,fps,durationA,baseOut,vidName):
    berstFeaturePerSec=[]
    berstFeaturePerSec2=[]
    berstTokensPerSec2=[]
    berstTokensPerSec=[]
    for i in range(len(allTextPerSec)):
        resultBert = bert_embedding(getAllTokenize(allTextPerSec[i]))
        resultBert2 = bert_embedding2nlp(" ".join(allTextPerSec[i]))
        tokenArr,featureArr=getFeatureArrayFromTokens(resultBert)
        print("at video :",vidName, " on sec: ",i)
        berstFeaturePerSec.append(featureArr)
        berstTokensPerSec.append(tokenArr)
        berstFeaturePerSec2.append(np.array(resultBert2))
        berstTokensPerSec2.append(" ".join(allTextPerSec[i]))
    return np.array(berstTokensPerSec),np.array(berstFeaturePerSec),np.array(berstTokensPerSec2),berstFeaturePerSec2

def generateTextFeaturesForAllSec3(bert_embedding,bert_embedding2nlp,allTextPerSec,fps,durationA,baseOut,vidName):
    berstFeaturePerSec=[]
    berstTokensPerSec=[]
    for i in range(len(allTextPerSec)):
        alltokens=getAllTokenize(allTextPerSec[i])
        
        resultBert = bert_embedding(alltokens)
        resultBert2 = bert_embedding2nlp(" ".join(allTextPerSec[i]))
        tokenArr,featureArr=getFeatureArrayFromTokens(resultBert)
        print("at video :",vidName, " on sec: ",i)
        berstFeaturePerSec.append(featureArr)
        berstTokensPerSec.append(tokenArr)
    return np.array(berstTokensPerSec),np.array(berstFeaturePerSec),np.array(resultBert2)


def combineStrList(strLs):
    finalStr=""
    for st in strls:
        if(st!="" and st!=" "):
            finalStr+=st

def getAllTokenize(lst):
    tokenize=[]
    for v in lst:
        if(v!="" and  v!=" " and v!='' and v!=' '):
            tokenize2=v.replace("\n"," ").split(" ")
            for tk in tokenize2:
                tkRefine=''.join(e for e in tk if e.isalnum())
                tokenize.append(tkRefine)
    return tokenize

def getFeatureArrayFromTokens(resultBert):
    tokenls=[]
    featureLs=[]
    for i in range(len(resultBert)):
        tokenls.append(resultBert[i][0])
        featureLs.append(resultBert[i][1])
    return np.array(tokenls),np.array(featureLs)
    


def getparts(oneLinerText,size=1600,ol=160):
    parts=[]
    poss=int(len(oneLinerText)/size)+1
    lag=0
    for i in range(poss):
        if(i>0):
            lag=ol
        parts.append(oneLinerText[i*size-lag:((i+1)*size)])
    return parts

import pandas as pd
# thsi is all names train + test files as in data directory
annotatedVideos=pd.read_csv('./annotatedData/annotatedVideos.csv',header=None,delimiter="XYZ").values 

base="./AnnotatedVideosAndSubtitles_version_1/"
basedirVids=base+"*.mp4"
basedirSubs="en_*.vtt"
baseOut="./data/textFeatures/"
baseDir=base
allVideos=glob.glob(basedirVids)

bert_embedding = BertEmbedding()
bert_embedding2nlp = pipeline('feature-extraction')

for v in range(len(allVideos)):
    vid=allVideos[v]
    vidName=vid.split("/")[-1][:-4]
    nameShowLimit=25
    if(not vidName in annotatedVideos): #check for already done videos
        print("video num: ",v," name: ",vidName[:nameShowLimit]," not found labeled")
        continue
    subs="en_"+vidName+".vtt"
    print("subs bert features for video: ",vidName, "idx: ",v)

    listAllText=getAllTextInSubs(baseDir+subs)
    oneLinerText=" ".join(listAllText)
    resultBertAll = bert_embedding(oneLinerText.replace("\n"," ").split(" "))
    tokenArrAll,featureArrAll=getFeatureArrayFromTokens(resultBertAll)
    partAll=getparts(oneLinerText)
    embedAllOuts2=[]
    textAllOut2=[]
    for i in range(len(partAll)):
        embedAllOuts2.append(np.array(bert_embedding2nlp(partAll[i])))
        textAllOut2.append(partAll[i])
    pickle.dump(tokenArrAll,open(baseOut+vidName+"_tokenArrAll.p","wb")) # token all
    pickle.dump(featureArrAll,open(baseOut+vidName+"_featureArrAll.p","wb")) # bert all
    pickle.dump(embedAllOuts2,open(baseOut+vidName+"_embedAllOuts2.p","wb"))# hugging face all
    pickle.dump(textAllOut2,open(baseOut+vidName+"_textAllOut2.p","wb"))# hugging face all
    print("features extracted now save as array: ",vidName, "idx: ",v)


