#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:30:23 2020

@author: ghauri
"""

from moviepy.editor import VideoFileClip
import numpy as np
import glob
import random

configDict={"REAL":0,"FAKE":1,"vidName":0,"label":1}

def getVidsLists(metaData,balance=True):
    dataList=[[],[]]# 0==real, 1==fake
    for k in metaData:
        dataList[configDict[metaData[k]["label"]]].append(k)
    if(balance):
        minV=min([len(dataList[configDict["REAL"]]),len(dataList[configDict["FAKE"]])])
        dataList[configDict["REAL"]]=random.sample(dataList[configDict["REAL"]], minV)
        dataList[configDict["FAKE"]]=random.sample(dataList[configDict["FAKE"]], minV)
    return dataList

def get1DVidLst(metaData,balance=True):
    dataList=getVidsLists(metaData,balance=True)
    trainDir=[]
    for i in range(len(dataList[configDict["REAL"]])):
        trainDir.append([dataList[configDict["REAL"]][i],configDict["REAL"]])
        trainDir.append([dataList[configDict["FAKE"]][i],configDict["FAKE"]])
    return trainDir

def getVidDirs(rootBase):
    vidDirs=glob.glob(rootBase)
    return vidDirs

def getMetaData(vidDir):
    metaData=eval(open(vidDir+"metadata.json","r").read())
    return metaData

def getVidFrames(vidName):
    video_clip = VideoFileClip(vidName)
    frmAll=list(video_clip.iter_frames(with_times=False))
    del video_clip
    return frmAll

def getVidFramesAndFpsDur(vidName):
    video_clip = VideoFileClip(vidName)
    frmAll=list(video_clip.iter_frames(with_times=False))
    frmAll,fps,duration=frmAll,video_clip.fps,video_clip.duration
    del video_clip
    return frmAll,fps,duration

def getVideoAudio(vidName):
    video_clip = VideoFileClip(vidName)
    audioclip = video_clip.audio
    del video_clip
    return audioclip

def getAudioFramesAndFpsDur(vidName):
    video_clip = VideoFileClip(vidName)
    audioclip = video_clip.audio
    frmAll=list(audioclip.iter_frames(with_times=False))
    frmAll,fps,duration=frmAll,audioclip.fps,audioclip.duration
    del video_clip
    return frmAll,fps,duration

