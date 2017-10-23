# encoding : utf-8
""" Python 2.7 """
""" Share Limitations GPLV3 """

import os
import math
import numpy as np
import random
from numpy.linalg import inv


b=None

def lr(TrainData,TrainMark,TestData,TestMark,epoch=10,fn=None):
    global b
    ltrain=len(TrainData)
    ltest=len(TestData)
    print(ltrain,ltest)
    w=len(TrainData[0])
    #print(b)
    ecnt=0
    alpha=1
    for i in xrange(epoch):
        der_1=np.matrix(np.zeros([w,1],dtype=np.float64))
        der_2=np.matrix(np.zeros([w,w],dtype=np.float64))
        for j in xrange(ltrain):
#           print(TrainData[j].shape,b.shape)
            ip=np.vdot(TrainData[j],b)
            thresh=10
            if ip>thresh:
                t1=1
                t2=0
            elif ip<-thresh:
                t1=0
                t2=0
            else:
                temp=np.exp(ip)
                t1=temp/(1+temp)
                t2=temp/((1+temp)*(1+temp))
            der_1=der_1+np.multiply((t1-TrainMark[j][0]),TrainData[j])
            der_2=der_2+np.multiply((TrainData[j]*TrainData[j].T),t2+0.001)
#        if np.linalg.cond(der_2)<1e5:
 #           der_2+=np.random.normal(size=(w,w),scale=0.001)
#        der_1+=alpha*2*b
#        der_2+=alpha*2
        dire=np.linalg.solve(der_2,der_1)
#        print dire
        b=b-dire*alpha
        alpha*=0.8
        #print(dire)
    cnt=0
    fp=open(fn,"wb")
    for j in xrange(ltest):

        temp=np.exp(np.vdot(TestData[j],b))
        p1=temp/(1+temp)
        if (p1>0.5):
            y=1
        else:
            y=0
        if y==TestMark[j][0]:
            cnt+=1
        fp.write("%d,%d\n"%(TestMark[j][1],y))
    fp.close()
    print("Accurate : %f " %( float(cnt)/ltest))


def KFold(DataList,DataMark,k,Trainer):
    global b
    l=len(DataList)
    assert(l>0)
    w=DataList[0].shape[0]
    lpart=l/k

    for i in xrange(k):
        #b=np.matrix(np.random.rand(w,1),dtype=np.float64)
        #b=b/(np.sqrt(np.vdot(b,b)))
        b=np.matrix(np.zeros([w,1]),dtype=np.float64)
        stp=i*lpart
        if (i==k-1):
            enp=l-1
        else:
            enp=(i+1)*lpart-1
        _TrainData=[]
        _TrainMark=[]
        _TestMark=[]
        _TestData=[]
        for j in xrange(l):
            if j>=stp and j<=enp:
                _TestData.append(DataList[j])
                _TestMark.append(DataMark[j])
            else:
                _TrainData.append(DataList[j])
                _TrainMark.append(DataMark[j])
        Trainer(_TrainData,_TrainMark,_TestData,_TestMark,100,"fold%d.csv"%(i+1))

def process():

# Read Data
    datalist=[]
    marklist=[]
    regdatalist=[]
    _rawdata=open("data.csv","rb").read().split()
    for i in _rawdata:
        i=i.split(",")
        for j in xrange(len(i)):
            i[j]=float(i[j])
        i.append(1)
        datalist.append(np.matrix(i,dtype=np.float64).T)
        #print(np.matrix(i,dtype=np.float64).T)

# Regularization
    l=len(datalist)
    mapping=range(l)
    random.shuffle(mapping)  
    w=datalist[0].shape[0]
    ex=np.zeros([w,1],dtype=np.float64)
    ex2=np.zeros([w,1],dtype=np.float64)
    for i in datalist:
       ex+=i
       ex2+=np.multiply(i,i)
    ex/=l
    ex2/=l
    var=np.sqrt(ex2-np.multiply(ex,ex))
    ex[w-1]=0
    var[w-1]=1
    #print("Variance : ",var)
    regdatalist=[0]*l
    marklist=[0]*l
    for i in xrange(l):
        regdatalist[mapping[i]]=(np.divide((datalist[i]-ex),var))

    _rawdata=open("targets.csv","rb").read().split()
    for i in xrange(l):
        marklist[mapping[i]]=(int(_rawdata[i]),i+1)
    KFold(regdatalist,marklist,10,lr);

if __name__=="__main__":
    print("Please Use Env Python 2.7")
    process()

