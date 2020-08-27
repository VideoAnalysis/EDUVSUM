

from src import utils, models
import json
import numpy as np
import math
import pickle
import time
from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import sys

def getOneHotClasses(gt,numClasses):
    oneHot=[]
    for i in range(len(gt)):
        curr=np.zeros(numClasses).astype(int)
        curr[int(gt[i]-1)]=1
        oneHot.append(curr)
    return np.array(oneHot)   

def getVisualFeatures(dataBase,visualFeatureTag,videoName,rsp,inShape):
#    featureDict={}
#    featureDict["vgg16Features"]=[7,7,512]
#    featureDict["inceptionV3Features"]=[8,8,2048]
#    featureDict["efficientNetB0Features"]=[7,7,1280]
    videoFreatures=pickle.load(open(dataBase+visualFeatureTag+"/"+videoName+"_"+visualFeatureTag+".p","rb"))
#    fts1=fts.reshape((videoFreatures.shape[0],featureDict[visualFeatureTag][0],featureDict[visualFeatureTag][1],featureDict[visualFeatureTag][2]))
    videoFreatures=videoFreatures.reshape((videoFreatures.shape[0],inShape[1],rsp))
    return np.mean(videoFreatures,axis=2)

#def getVisualFeatures2(dataBase,visualFeatureTag,videoName,rsp,inShape):
    
def getCombinedTxtFeatures(textFreaturesAll2):
    textFreaturesAll2[0]=textFreaturesAll2[0].reshape((textFreaturesAll2[0].shape[1],textFreaturesAll2[0].shape[2]))
    stackedAll=textFreaturesAll2[0]#textFreaturesAll2[0][:-int(textFreaturesAll2[0].shape[0]*0.10)]
    for i in range(1,len(textFreaturesAll2)):
        textFreaturesAll2[i]=textFreaturesAll2[i].reshape((textFreaturesAll2[i].shape[1],textFreaturesAll2[i].shape[2]))
        part=textFreaturesAll2[i][int(textFreaturesAll2[i].shape[0]*0.10):]#-int(textFreaturesAll2[i].shape[0]*0.10)
        stackedAll=np.concatenate([stackedAll,part],axis=0)  
    return stackedAll

def getCorreshist(hist,feature,featureDim,vidFps,vidFrs):
    ttime=vidFrs/vidFps
    return math.floor(featureDim/ttime)
#    if(feature=="v"):
#        return math.floor(featureDim//ttime)
#    if(feature=="t"):
#        return featureDim//ttime
#    if(feature=="a"):
#        return featureDim//ttime


def getClassesFromOneHot(y,yHat):
    yc=[]
    yHatc=[]
    for i in range(len(y)):
        yc.append(np.argmax(y[i]))
        yHatc.append(np.argmax(yHat[i]))
    return np.array(yc),np.array(yHatc)

def getAccuracy(y,yHat):
    accCnt=0
    for i in range(len(y)):
        if(y[i]==yHat[i]):
            accCnt=+1
    total=len(y)
    if(len(y)==0):
        total=999999999999
    acc=accCnt/total
    return acc

def getUniformSampled(visualFeatures,otherFeatures,vidFps,mod=""):
    nm=visualFeatures.shape[0]//vidFps
    anm=otherFeatures.shape[0]//nm
    asmpr=anm*nm
    if(mod=="a"):
        asmpr=visualFeatures.shape[0]*10
    if(mod=="t"):
        asmpr=visualFeatures.shape[0]
    otherFeaturesArange=np.arange(otherFeatures.shape[0])
    otherFeaturesSmp=otherFeatures[np.random.choice(otherFeaturesArange,asmpr)]
    return otherFeaturesSmp

def getTestResults(model,allFilesTvSumTest,histV,histA,histT,gtbase,modelName,params):
    testMetricsout=[]
    yNyhatByVideo=[]
    testMaeLossLs=[]
    tMseLs=[]
    accLs=[]
    prcLs=[]
    recLs=[]
    fscoreLs=[]
    allMetric=[]
    vidFps=params["vidFps"]
    visualIndicator=params["visualIndicator"]
    audioIndicator=params["audioIndicator"]
    textIndicator=params["textIndicator"]
    for j in range(len(allFilesTvSumTest)):
#        visualFeatures=getVisualFeatures(dataBase,visualFeatureTag,allFilesTvSumTest[j],rsp,inShape)
        visualFeatures=pickle.load(open(dataBase+visualFeatureTag+"/"+allFilesTvSumTest[j]+"_"+visualFeatureTag+".p","rb"))
        visualFeatures=visualFeatures.reshape((visualFeatures.shape[0],visualFeatures.shape[2]))
        gt=pickle.load(open(gtbase+allFilesTvSumTest[j]+gtpostfix,"rb"))
        gtoH=getOneHotClasses(gt,numClasses)
        audioFeatures=pickle.load(open(dataBase+audioFeatureTag+"/"+allFilesTvSumTest[j]+"_0.05_0.025_"+audioFeatureTag+".p","rb"))
        audioFeatures[0]=audioFeatures[0].reshape((audioFeatures[0].shape[1],audioFeatures[0].shape[0]))
        audFeatures=audioFeatures[0]
        audFeatures= getUniformSampled(visualFeatures,audFeatures,vidFps,mod="a")
        
        textFreaturesAll2=pickle.load(open(dataBase+textFeatureTag+"/"+allFilesTvSumTest[j]+"_"+textFeatureTag+".p","rb"))
        txtFeatures=getCombinedTxtFeatures(textFreaturesAll2)
        txtFeatures= getUniformSampled(visualFeatures,txtFeatures,vidFps,mod="t")
        yhatLs=[]
        Y=[]
        maxPoss=(visualFeatures.shape[0]//vidFps)*vidFps
        batch=1
#        maxPoss=(maxPoss//(histV*batch))*(histV*batch)
#        yOrg=0
        for h in range(histV,maxPoss,histV):
            xtest=[]
            b=0
            vInputbatch=[]
            aInputbatch=[]
            tInputbatch=[]
            for b in range(batch):
                if(h+b-1>=len(gtoH)):
                    break
                if(visualIndicator in modelName):
                    vInput=visualFeatures[h+b-histV:h+b]
                    vInput=vInput.reshape((1,vInput.shape[0],vInput.shape[1]))
                    vInputbatch.append(vInput)
                if(audioIndicator in modelName):
                    aInput=audFeatures[((h//histV)*histA)+b-histA:((h//histV)*histA)+b]
                    aInput=aInput.reshape((1,aInput.shape[0],aInput.shape[1]))
                    aInputbatch.append(aInput)
                if(textIndicator in modelName):
                    tInput=txtFeatures[((h//histV)*histT)+b-histT:((h//histV)*histT)+b]
                    tInput=tInput.reshape((1,tInput.shape[0],tInput.shape[1]))
                    tInputbatch.append(tInput)
            if(len(vInputbatch)==0 and len(aInputbatch)==0 and len(tInputbatch)==0):
                break
            if(visualIndicator in modelName):
                vInputbatch=np.array(vInputbatch).reshape((batch,vInput.shape[1],vInput.shape[2]))
            if(audioIndicator in modelName):
                aInputbatch=np.array(aInputbatch).reshape((batch,aInput.shape[1],aInput.shape[2]))
            if(textIndicator in modelName):
                tInputbatch=np.array(tInputbatch).reshape((batch,tInput.shape[1],tInput.shape[2]))
            allInputs=[vInputbatch,aInputbatch,tInputbatch]
            xtest=[x for x in allInputs if len(x)>0]
            if(h+b-1>=len(gtoH) or len(xtest)==0):
                break
            yOrg=gtoH[h+b-1]
            yhat=model.predict(xtest)
            yhatLs.append(yhat[0])
            
            Y.append(yOrg)
    #    print("test: ",j," name: ",allFilesTvSumTest[j])
        y_true, y_pred=getClassesFromOneHot(Y,yhatLs)
        tMse=(np.square(np.array(y_true) - np.array(y_pred))).mean(axis=0)
        tMseLs.append(tMse)
        acc=getAccuracy(y_true, y_pred)
        accLs.append(acc)
        macRes=precision_recall_fscore_support(y_true, y_pred, average='macro')
        micRes=precision_recall_fscore_support(y_true, y_pred, average='micro')
        wRes=precision_recall_fscore_support(y_true, y_pred, average='weighted')
        pS=spearmanr(np.array(y_true),np.array(y_pred))[0]
        kT=kendalltau(rankdata(-np.array(y_true)), rankdata(-np.array(y_pred)))[0]
        allMetric.append([tMse,acc,macRes[0],macRes[1],macRes[2],micRes[0],micRes[1],micRes[2],wRes[0],wRes[1],wRes[2],kT,pS])
        yNyhatByVideo.append([allFilesTvSumTest[j],Y,yhatLs])
    return allMetric,yNyhatByVideo


#params=json.load(open("parameters.json","r"))
print("all args on console: ",sys.argv)
params=[]
if(len(sys.argv)>1):
    params=json.load(open(sys.argv[1],"r"))
else:
    params=json.load(open("parameters.json","r"))


#tvSumBase="./data/"+utils.getFileName(params["visualFeatureTag"])[0]+"/"
#vfbase="_"+utils.getFileName(params["visualFeatureTag"])[1]+".p" 
gtbase=params["gtbase"]#"./data/gtAvg/"
gtpostfix=params["gtpostfix"]#"_smpGt.p"

baseout=params["output"]

dataBase=params["dataBase"]#"./data/"
#textTag="bertFeaturesPerSec"
#textTag2="bertFeaturesPerSec2"
#textAll1="featureArrAll"
#textAll2="embedAllOuts2"
#textTag="embedAllOuts2"

#audioTag1="featuresAudio"
#audioTag2="sampledFramesAudio"
#videoTagEff="efficientNetB0Features"
#videoTagIncep="InceptionV3Features"
#videoTagVgg="vgg16Features"

#visualFeatureTag=videoTagVgg
#audioFeatureTag=audioTag1
#textFeatureTag=textTag


#allFilesTvSum=utils.getVideoList(params["videoList"])

#audioFeatures=pickle.load(open(dataBase+audioFeatureTag+"/"+allFilesTvSum[0]+"_0.05_0.025_"+audioFeatureTag+".p","rb"))
#audioFeatures[0]=audioFeatures[0].reshape((audioFeatures[0].shape[1],audioFeatures[0].shape[0]))
##  audioFeatures[0].shape (68, 5177) #  len(audioFeatures[1]) name of features
#
#audioFeatures2=pickle.load(open(dataBase+audioTag2+"/"+allFilesTvSum[0]+"_4410_"+audioTag2+".p","rb"))
##(568890, 2)   for 129 secs
#
#textFreatures=pickle.load(open(dataBase+textTag+"/"+allFilesTvSum[0]+"_"+textTag+".p","rb"))
#textTokensFreatures=pickle.load(open(dataBase+textTag+"/"+allFilesTvSum[0]+"_"+"bertTokensPerSec"+".p","rb"))
## (129,) (129, 23, 1, 768)
#
#textFreatures2huggingFace=pickle.load(open(dataBase+textTag2+"/"+allFilesTvSum[0]+"_"+"bertFeaturesPerSec2"+".p","rb"))

#textFreaturesAll1=pickle.load(open(dataBase+textAll1+"/"+allFilesTvSum[0]+"_"+textAll1+".p","rb"))
#textFreaturesAll2=pickle.load(open(dataBase+textFeatureTag+"/"+allFilesTvSum[0]+"_"+textFeatureTag+".p","rb"))


#videoFreaturesVgg=pickle.load(open(dataBase+videoTagVgg+"/"+allFilesTvSum[0]+"_"+videoTagVgg+".p","rb"))
#videoFreaturesVgg.shape
##(387, 1, 7, 7, 512)
#audioFeaturesIncep=pickle.load(open(dataBase+videoTagIncep+"/"+allFilesTvSum[0]+"_"+videoTagIncep+".p","rb"))
#audioFeaturesIncep.shape
##(387, 1, 8, 8, 2048)
#audioFeaturesEff=pickle.load(open(dataBase+videoTagEff+"/"+allFilesTvSum[0]+"_"+videoTagEff+".p","rb"))
#audioFeaturesEff=audioFeaturesEff.reshape((audioFeaturesEff.shape[0],1,7,7,1280))
#audioFeaturesEff.shape
## (387, 1280, 7, 7)

#smp1=np.zeros((2,3))
#smp2=np.zeros((2,3))
#np.vstack([smp1,smp2]).shape
#np.hstack([smp1,smp2]).shape

histLs=params["histLs"]
#hist=1
#rsp,fshape,inShape=utils.getFeatureShape(params["visualFeatureTag"],hist)
#inShape=(hist,fshape)

epochs=params["epochs"]
#epochs=5
savePoint=epochs-1

batch=params["batch"]
#batch=2
data=params["dataName"]#"TvSum"

#vFeatureMod=params["visualFeatureTag"]
#aFeatureMod=params["visualFeatureTag"]
#tFeatureMod=params["visualFeatureTag"]



optimizerNameLs=params["optimizerNameLs"]#["adam"]
activationNameLs=params["activationNameLs"]#["softmax"] #sigmoid #softmax



totalMod=len(optimizerNameLs)*len(activationNameLs)
modeCount=0
#
#trainRatio=0.80
#trainidx=math.ceil(trainRatio*len(allFilesTvSum))
#
#allFilesTvSumTrain=allFilesTvSum[:trainidx]
#allFilesTvSumTest=allFilesTvSum[trainidx:]
allFilesTvSumTrain=utils.getVideoList(params["trainVideoList"])
allFilesTvSumTest=utils.getVideoList(params["testVideoList"])




eta=params["eta"] #0.00000001
stopCount=params["stopCount"] #5

lsAllTrainMse0=[]
lsAllTrainMse1=[]
lsAllyNyhatbyvideo=[]
lsAllTestMse=[]
lsAllTestLsMse=[]
lsAllTime=[]
paramLs=[]
maeByVidLs=[]

acth=params["activationHidden"]#"tanh" #relu # tanh
actf=params["activationFinal"]

gt=pickle.load(open(gtbase+allFilesTvSumTrain[0]+gtpostfix,"rb"))
numClasses=params["numClasses"]
gtoH=getOneHotClasses(gt,numClasses)


#allFilesTvSumTest=allFilesTvSumTrain[:2]
vidFps=params["vidFps"]
visualIndicator=params["visualIndicator"]
audioIndicator=params["audioIndicator"]
textIndicator=params["textIndicator"]
visualFeatureTags=params["visualFeatureTags"]
audioFeatureTags=params["audioFeatureTags"]
textFeatureTags=params["textFeatureTags"]


modelNames=params["models"]

#for actf in activationNameLs:
#        model=models.getModel(modelName,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth)
 #                    visualFeatures=getVisualFeatures(dataBase,visualFeatureTag,allFilesTvSumTrain[j],rsp,inShape)
 
for modelName in modelNames:
    for visualFeatureTag in visualFeatureTags:
        for audioFeatureTag in audioFeatureTags:
            for textFeatureTag in textFeatureTags:
                #visualFeatures=getVisualFeatures(dataBase,visualFeatureTag,allFilesTvSumTrain[0],rsp,inShape)
                visualFeatures=pickle.load(open(dataBase+visualFeatureTag+"/"+allFilesTvSumTrain[0]+"_"+visualFeatureTag+".p","rb"))
#                visualFeatures=pickle.load(open("/home/ghauri/Documents/Tib_drive/tib_main/git_dev/lectureVideoSummary/multiModalSourceCode/data/vgg16Features/Advanced_CPU_Designs__Crash_Course_Computer_Science_9_-_English_csyt_vgg16Features.p","rb"))
                visualFeatures=visualFeatures.reshape((visualFeatures.shape[0],visualFeatures.shape[2]))
                
                
                

                #audioFeatures[0]
                #textFreaturesAll2
                audioFeatures=pickle.load(open(dataBase+audioFeatureTag+"/"+allFilesTvSumTrain[0]+"_0.05_0.025_"+audioFeatureTag+".p","rb"))
                audioFeatures[0]=audioFeatures[0].reshape((audioFeatures[0].shape[1],audioFeatures[0].shape[0]))
                audFeatures=audioFeatures[0]
                audFeatures= getUniformSampled(visualFeatures,audFeatures,vidFps,mod="a")
                
                #txtFeatures=np.ones((visualFeatures.shape[0],1,68))
                textFreaturesAll2=pickle.load(open(dataBase+textFeatureTag+"/"+allFilesTvSumTrain[0]+"_"+textFeatureTag+".p","rb"))
                txtFeatures=textFreaturesAll2
                txtFeatures=getCombinedTxtFeatures(textFreaturesAll2)
                txtFeatures= getUniformSampled(visualFeatures,txtFeatures,vidFps,mod="t")
                for hist in histLs:
                    histV=getCorreshist(hist,feature="v",featureDim=visualFeatures.shape[0],vidFps=vidFps,vidFrs=visualFeatures.shape[0])
                    histA=getCorreshist(hist,feature="a",featureDim=audFeatures.shape[0],vidFps=vidFps,vidFrs=visualFeatures.shape[0])
                    histT=getCorreshist(hist,feature="t",featureDim=txtFeatures.shape[0],vidFps=vidFps,vidFrs=visualFeatures.shape[0])
                    
                    inputShapeV=[histV,visualFeatures.shape[1]]
                    inputShapeA=[histA,audFeatures.shape[-1]]
                    inputShapeT=[histT,txtFeatures.shape[-1]]
    
                    for opt in optimizerNameLs:
                        modeCount+=1
                        model=models.getModel(params,modelName,inputShapeV,inputShapeT,inputShapeA,opt,actf,acth,numClasses)
                
                        lossEpMaeMse=[[],[]]
                        fscoreTestLs=[]
                        timeTakenLs=[]
                        ls=[]
                        batch=params["batch"]
                        modelType=modelName+"_" +data+"_"+str(epochs)+"_"+str(hist)+"_"+str(batch)+"_"+visualFeatureTag+"_"+audioFeatureTag+"_"+textFeatureTag+"_"+opt+"_"+actf+"_"+acth
                        paramLs.append([modelName,data,str(epochs),str(hist),str(batch),visualFeatureTag,audioFeatureTag,textFeatureTag,opt,actf])
                        print("now in model :",modelType," :",modeCount," out: ",totalMod)
                        print("***************************************************")
                        epCnt=0
                        maxAcc=0
                        for i in range(epochs):
                            strTime=time.time()
                            lossLsMaeMse=[[],[]]
                            lossLsMae=[]
                            for j in range(len(allFilesTvSumTrain)):
                                gt=pickle.load(open(gtbase+allFilesTvSumTrain[j]+gtpostfix,"rb"))
                                gtoH=getOneHotClasses(gt,numClasses)
    #                            if(visualIndicator in modelName):
                                visualFeatures=pickle.load(open(dataBase+visualFeatureTag+"/"+allFilesTvSumTrain[j]+"_"+visualFeatureTag+".p","rb"))
                                visualFeatures=visualFeatures.reshape((visualFeatures.shape[0],visualFeatures.shape[2]))
                                if(audioIndicator in modelName):
                                    audioFeatures=pickle.load(open(dataBase+audioFeatureTag+"/"+allFilesTvSumTrain[j]+"_0.05_0.025_"+audioFeatureTag+".p","rb"))
                                    audioFeatures[0]=audioFeatures[0].reshape((audioFeatures[0].shape[1],audioFeatures[0].shape[0]))
                                    audFeatures=audioFeatures[0]
                                    audFeatures= getUniformSampled(visualFeatures,audFeatures,vidFps,mod="a")
                                if(textIndicator in modelName):
                                    txtFeatures=pickle.load(open(dataBase+textFeatureTag+"/"+allFilesTvSumTrain[j]+"_"+textFeatureTag+".p","rb"))
                                    txtFeatures=getCombinedTxtFeatures(txtFeatures)
                                    txtFeatures= getUniformSampled(visualFeatures,txtFeatures,vidFps,mod="t")
                                maxPoss=(visualFeatures.shape[0]//vidFps)*vidFps
#                                maxPoss=(maxPoss//(histV*batch))*(histV*batch)
                                for h in range(histV,maxPoss,histV*batch):#len(gt)
                                    xtrain=[]
                                    ytrain=[]
                                    b=0
                                    vInputbatch=[]
                                    aInputbatch=[]
                                    tInputbatch=[]
                                    remain=visualFeatures.shape[0]-h
                                    if(remain<=batch):
                                        batch=remain
                                    for b in range(batch):
    #                                    if(visualIndicator in modelName):
    #                                        vInput=visualFeatures[h+b-histV:h+b]
    #                                        vInput=vInput.reshape((1,vInput.shape[0],vInput.shape[1]))
    #                                        vInputbatch.append(vInput)
    #                                    if(audioIndicator in modelName):
    #                                        aInput=audFeatures[((h//histV)*histA)+b-histA:((h//histV)*histA)+b]
    #                                        aInput=aInput.reshape((1,aInput.shape[0],aInput.shape[1]))
    #                                        aInputbatch.append(aInput)
    #                                    if(textIndicator in modelName):
    #                                        tInput=txtFeatures[((h//histV)*histT)+b-histT:((h//histV)*histT)+b]
    #                                        tInput=tInput.reshape((1,tInput.shape[0],tInput.shape[1]))
    #                                        tInputbatch.append(tInput)
                                        if(h+b-1>=len(gtoH)):
                                            break
                                        if(visualIndicator in modelName):
                                            vInput=visualFeatures[h+b-histV:h+b]
                                            vInput=vInput.reshape((1,vInput.shape[0],vInput.shape[1]))
                                            vInputbatch.append(vInput)
                                        if(audioIndicator in modelName):
                                            aInput=audFeatures[((h//histV)*histA)+b-histA:((h//histV)*histA)+b]
                                            aInput=aInput.reshape((1,aInput.shape[0],aInput.shape[1]))
                                            aInputbatch.append(aInput)
                                        if(textIndicator in modelName):
                                            tInput=txtFeatures[((h//histV)*histT)+b-histT:((h//histV)*histT)+b]
                                            tInput=tInput.reshape((1,tInput.shape[0],tInput.shape[1]))
                                            tInputbatch.append(tInput)
                                        
                                        ytrain.append(gtoH[h+b-1])#[h+b-3]
                                    if(visualIndicator in modelName):
                                        vInputbatch=np.array(vInputbatch).reshape((len(vInputbatch),vInput.shape[1],vInput.shape[2]))
                                    if(audioIndicator in modelName):
                                        aInputbatch=np.array(aInputbatch).reshape((len(aInputbatch),aInput.shape[1],aInput.shape[2]))
                                    if(textIndicator in modelName):
                                        tInputbatch=np.array(tInputbatch).reshape((len(tInputbatch),tInput.shape[1],tInput.shape[2]))
                                    allInputs=[vInputbatch,aInputbatch,tInputbatch]
                                    if(len(ytrain)==0):
                                        break
                                    xtrain=[x for x in allInputs if len(x)>0]
                    #                    ytrain=np.array([[gt[h+b-1]]])
                                    ytrainB=np.array(ytrain) # cause index -1 which here 1=3
                                    #xtrain=utils.getReshape(xtrain,inShape,modelName,batch)
                                    ls=model.train_on_batch(xtrain,ytrainB)
                                    lossLsMaeMse[0].append(ls[0])
                                    lossLsMaeMse[1].append(ls[1])
                            lossEpMaeMse[0].append(np.mean(lossLsMaeMse[0]))
                            lossEpMaeMse[1].append(np.mean(lossLsMaeMse[1]))
                            if(i>3 and (lossEpMaeMse[0][-1]-lossEpMaeMse[0][-2])<eta):
                                epCnt+=1
                            else:
                                epCnt=0
                            if(lossEpMaeMse[1][-1]>=maxAcc):
                                maxAcc=lossEpMaeMse[1][-1]
                                model.save(baseout+modelType+"_modelMxScr_whole.h5")
                            if(epCnt>=stopCount):
                                model.save(baseout+modelType+"_model_whole.h5")
                                break
                            print("train at epoch: ",i," loss:",[lossEpMaeMse[0][-1],lossEpMaeMse[1][-1]], "timetaken: ",utils.getTimeElapsed(strTime)[0])
                            if(i%savePoint==0 and i>0):
                                model.save(baseout+modelType+"_model_whole.h5")
                            timeTakenLs.append(utils.getTimeElapsed(strTime)[0])
                        allMetric,yNyhatByVideo=getTestResults(model,allFilesTvSumTest,histV,histA,histT,gtbase,modelName,params)
                        allMetricArr=np.array(allMetric)
                        outName=modelType
                        np.savetxt(baseout+outName+"_allMetric.csv", np.array(allMetric), delimiter=",", fmt='%s')
                        #        np.savetxt(baseout+outName+"_allMetric.csv", np.array(allMetric), delimiter=",", fmt='%s')
                        pickle.dump(np.array([lsAllTrainMse0,lsAllTrainMse1]),open(baseout+outName+"_lsAllTrainloss.p","wb"))
                        pickle.dump(np.array(allMetric),open(baseout+outName+"_allMetric.p","wb"))
                        pickle.dump(np.array(yNyhatByVideo),open(baseout+outName+"_yNyhatByVideo.p","wb"))


    
#accForAllvids=[]
#for j in range(len(yNyhatByVideo)):
#    sameCnt=0
#    top=2
#    for i in range(len(yNyhatByVideo[j][1])):
#        arsrt=np.argsort(yNyhatByVideo[j][2][i])
#        #np.argmax(yNyhatByVideo[0][2][i])
#        if(np.argmax(yNyhatByVideo[j][1][i]) in arsrt[-top:]):
#            sameCnt+=1
#            
#    tcnt=len(yNyhatByVideo[j][1])    
#    accThis=sameCnt/tcnt
#    accForAllvids.append(accThis)
#    
#accForAllvids
#np.mean(accForAllvids)
#np.max(accForAllvids)
##
#pickle.dump(np.array(yNyhatByVideo),open(baseout+outName+"_yNyhatByVideo_special2.p","wb"))
#nm=visualFeatures.shape[0]//3
#
#anm=audioFeatures[0].shape[0]//nm
#asmpr=anm*nm
#audioFeaturesArange=np.arange(audioFeatures[0].shape[0])
#audioFeaturesSmp=audioFeatures[0][np.random.choice(audioFeaturesArange,asmpr)]
#
#tnm=txtFeatures.shape[0]//nm
#tsmpr=tnm*nm
#
#audioFeaturesArange=np.arange(audioFeatures[0].shape[0])
#audioFeaturesSmp=audioFeatures[0][np.random.choice(audioFeaturesArange,asmpr)]
#
#audioFeatures
#txtFeatures

#vInputbatch=[1]
#aInputbatch=[]
#tInputbatch=[]
#allInputs=[vInputbatch,aInputbatch,tInputbatch]
#
#xtrain=[x for x in allInputs if len(x)>0]

#import pandas as pd
#pickle.dump(np.array(yNyhatByVideo),open("test_vInput.p","wb"))
#
#import h5py
#
#dt = h5py.string_dtype(encoding='ascii')
#dt = h5py.string_dtype()
#
#
#test2=pickle.load(open("test2.p","rb"))
#
#with h5py.File('test2.h5', 'w') as hf:
#    hf.create_dataset("test2",  data=test2) #, dtype=dt
#
#np.save(open('test2.npy', 'wb'),test2)
#
#test2Arr=np.load(open('test2.npy', 'rb'))
#
#import feather
#
#df = pd.DataFrame(vInput)
#
#feather.write_dataframe(vInput, 'test_vInput.feather')
    
#allFilesTvSumTest=allFilesTvSumTrain[:2]

#testMetricsout=[]
#yNyhatByVideo=[]
#testMaeLossLs=[]
#tMseLs=[]
#accLs=[]
#prcLs=[]
#recLs=[]
#fscoreLs=[]
#allMetric=[]
#for j in range(len(allFilesTvSumTest)):
#    visualFeatures=getVisualFeatures(dataBase,visualFeatureTag,allFilesTvSumTest[j],rsp,inShape)
#    visualFeatures=visualFeatures.reshape((visualFeatures.shape[0],visualFeatures.shape[1]))
#    gt=pickle.load(open(gtbase+allFilesTvSumTest[j]+gtpostfix,"rb"))
#    gtoH=getOneHotClasses(gt,numClasses)
#    audioFeatures=pickle.load(open(dataBase+audioTag1+"/"+allFilesTvSumTest[j]+"_0.05_0.025_"+audioTag1+".p","rb"))
#    audioFeatures[0]=audioFeatures[0].reshape((audioFeatures[0].shape[1],audioFeatures[0].shape[0]))
#    audFeatures=audioFeatures[0]
#    textFreaturesAll2=pickle.load(open(dataBase+textAll2+"/"+allFilesTvSumTest[j]+"_"+textAll2+".p","rb"))
#    txtFeatures=getCombinedTxtFeatures(textFreaturesAll2)
#    yhatLs=[]
#    Y=[]
#    for h in range(histV,len(gt),histV):
#        xtest=[]
#        b=0
#        vInputbatch=[]
#        aInputbatch=[]
#        tInputbatch=[]
#        batch=1
#        for b in range(batch):
#            vInput=visualFeatures[h+b-histV:h+b]
#            vInput=vInput.reshape((1,vInput.shape[0],vInput.shape[1]))
#            aInput=audFeatures[((h//histV)*histA)+b-histA:((h//histV)*histA)+b]
#            aInput=aInput.reshape((1,aInput.shape[0],aInput.shape[1]))
#            tInput=txtFeatures[((h//histV)*histT)+b-histT:((h//histV)*histT)+b]
#            tInput=tInput.reshape((1,tInput.shape[0],tInput.shape[1]))
#            vInputbatch.append(vInput)
#            aInputbatch.append(aInput)
#            tInputbatch.append(tInput)
#            ytrain.append(gtoH[h+b])#[h+b-3]
#        vInputbatch=np.array(vInputbatch).reshape((batch,vInput.shape[1],vInput.shape[2]))
#        aInputbatch=np.array(aInputbatch).reshape((batch,aInput.shape[1],aInput.shape[2]))
#        tInputbatch=np.array(tInputbatch).reshape((batch,tInput.shape[1],tInput.shape[2]))
#        xtest=[vInputbatch,aInputbatch,tInputbatch]
#
#        yhat=model.predict(xtest)
#        yhatLs.append(yhat[0])
#        yOrg=gtoH[h+b]
#        Y.append(yOrg)
##    print("test: ",j," name: ",allFilesTvSumTest[j])
#    y_true, y_pred=getClassesFromOneHot(Y,yhatLs)
#    tMse=(np.square(np.array(y_true) - np.array(y_pred))).mean(axis=0)
#    tMseLs.append(tMse)
#    acc=getAccuracy(y_true, y_pred)
#    accLs.append(acc)
#    macRes=precision_recall_fscore_support(y_true, y_pred, average='macro')
#    micRes=precision_recall_fscore_support(y_true, y_pred, average='micro')
#    wRes=precision_recall_fscore_support(y_true, y_pred, average='weighted')
#    pS=spearmanr(np.array(y_true),np.array(y_pred))[0]
#    kT=kendalltau(rankdata(-np.array(y_true)), rankdata(-np.array(y_pred)))[0]
#    allMetric.append([tMse,acc,macRes[0],macRes[1],macRes[2],micRes[0],micRes[1],micRes[2],wRes[0],wRes[1],wRes[2],kT,pS])
#    yNyhatByVideo.append([allFilesTvSumTest[j],Y,yhatLs])
#return allMetric,yNyhatByVideo
#        lsAllTrainMse0.append(lossEpMaeMse[0])
#        lsAllTrainMse1.append(lossEpMaeMse[1])
#        lsAllyNyhatbyvideo.append(yNyhatByVideo)
#        lsAllTestMse.append(testLoss)
#        lsAllTestLsMse.append(testLossLs)
#        lsAllTime.append(sum(timeTakenLs))
#        print("Last testLoss is: ",lsAllTestMse[-1])
#        print("total time taken is: ",sum(timeTakenLs))

#outName=modelName+"_" +data+"_"+str(epochs)+"_"+str(hist)+"_"+str(batch)+"_"+featureMod
#np.savetxt(baseout+outName+"_paramLs.csv", np.array(paramLs), delimiter=",", fmt='%s')
#pickle.dump(np.array([lsAllTrainMse0,lsAllTrainMse1]),open(baseout+outName+"_lsAllTrainloss.p","wb"))
#pickle.dump(np.array(lsAllyNyhatbyvideo),open(baseout+outName+"_lsAllyNyhatbyvideo.p","wb"))
#np.savetxt(baseout+outName+"_lsAllTestMse.csv", np.array(lsAllTestMse), delimiter=",", fmt='%s')
#pickle.dump(np.array(lsAllTestLsMse),open(baseout+outName+"_lsAllTestLsMse.p","wb"))
#np.savetxt(baseout+outName+"_lsAllTime.csv", np.array(lsAllTime), delimiter=",", fmt='%s')


    
#y_true, y_pred=getClassesFromOneHot(Y,yhatLs)
#    
#from sklearn.metrics import classification_report
#y_true = [0, 1, 2, 2, 2]
#y_pred = [0, 0, 2, 2, 1]
#
#target_names = ['class 0', 'class 1', 'class 2']
#res=classification_report(y_true, y_pred)
#
#print(res)
#
#import numpy as np
#from sklearn.metrics import precision_recall_fscore_support
#y_true = [0, 1, 2, 2, 2]
#y_pred = [0, 0, 2, 2, 1]
#hat=np.array(y_true)^np.array(y_pred)
#
#def getMseClasses(y,yHat):
#    
#    
#    
#macRes=precision_recall_fscore_support(y_true, y_pred, average='macro')
#micRes=precision_recall_fscore_support(y_true, y_pred, average='micro')
#wRes=precision_recall_fscore_support(y_true, y_pred, average='weighted')



