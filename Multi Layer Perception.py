import os
import numpy as np

def read(datapath,labelpath):
    fp=open(datapath,"r")
    _data=[]
    for i in fp:
        i=i.split(",")
        _item=[]
        for j in i:
            _item.append(float(j))
        #_item.append(1.0)
        _data.append(np.array(_item,dtype=np.float64))
    fp=open(labelpath,"r")
    _label=[]
    for i in fp:
        i=i.split(",")
        _item=[]
        for j in i:
            _item.append(int(j))
        _label.append(_item)
    return _data,_label

def create_net(mlp_shape,init='zero'):
    l=len(mlp_shape)
    w=[]
    b=[]
    for i in range(l-1):
        w.append(np.random.rand(mlp_shape[i],mlp_shape[i+1])/20)
        b.append(np.zeros([1,mlp_shape[i+1]]))
    net=[w[0],w[1],b[0],b[1]]
    return net

def forward(net,data):
    w1=net[0]
    w2=net[1]
    b1=net[2]
    b2=net[3]
    mark=[]
    l1=np.add(np.dot(data,w1),b1)
    a1=np.divide(1,np.add(1,np.exp(l1)))
    l2=np.add(np.dot(a1,w2),b2)
    a2=np.divide(1,np.add(1,np.exp(l2)))
    return a2

def fcerror(w,loss):
    """
        w il*ol
        loss row vector
    """
    ol=w.shape[1]
    il=w.shape[0]
    error=[0]*w[0]
    """
    for i in range(il):
        for j in range(ol):
            error[i]+=w[i][j]*loss[j]
    """
    error=np.dot(loss,w.T)
    return error

def sigmoidloss(x,loss):
    return -x*(1-x)*loss

def learn(net,data,label,learning_rate=0.1):
    w1=net[0]
    w2=net[1]
    b1=net[2]
    b2=net[3]
    l=len(data)
    lr=learning_rate
    _mse=0
    _deltaw1=0
    _deltaw2=0
    _deltab1=0
    _deltab2=0
    gamma=0.1
    batch=2
    for i in range(l):
        _sample=data[i].reshape([1,-1])
        _label=label[i]
        #print ("w",w1.shape,"b1",b1.shape)
        l1=np.add(np.dot(_sample,w1),b1)
        #print ("l1",l1.shape)
        a1=np.divide(1,np.add(1,np.exp(l1)))
        #print ("a1",a1.shape)
        l2=np.add(np.dot(a1,w2),b2)
        #print ("l2",l2.shape)
        a2=np.divide(1,np.add(1,np.exp(l2)))        
        #print ("a2",a2.shape)
        _labeloh=np.zeros([1,w2.shape[1]])
        _labeloh[0][_label[0]]=1
        _gloss=-(a2-_labeloh)
        
        _mse+=np.sum(_gloss*_gloss,axis=1)
        _ga2=sigmoidloss(a2,_gloss)
        _gl2=fcerror(w2,_ga2)
        _ga1=sigmoidloss(a1,_gl2)
        _gl1=fcerror(w2,_ga2)
        _deltaw2+=np.dot(a1.T,_ga2)
        _deltaw1+=np.dot(_sample.T,_ga1)
        _deltab2+=_ga2
        _deltab1+=_ga1
        if i%batch==0:
            w1+=lr/batch*_deltaw1
            w2+=lr/batch*_deltaw2
            b1+=lr/batch*_deltab1
            b2+=lr/batch*_deltab2
            _deltaw1=0
            _deltaw2=0
            _deltab1=0
            _deltab2=0
    w1+=lr*_deltaw1
    w2+=lr*_deltaw2
    b1+=lr*_deltab1
    b2+=lr*_deltab2
    _mse/=l
    return [w1,w2,b1,b2],_mse

if __name__=="__main__":
    _traindata,_trainlabel=read("train_data.csv","train_targets.csv")
    _testdata,_testlabel=read("test_data.csv","train_targets.csv")
    net=create_net([400,100,10])
    for i in range(20):
        net,mse=learn(net,_traindata,_trainlabel)
        print (mse)
    _result=forward(net,_testdata)
    _result=np.argmax(_result,axis=1)
    f=open("test_predictions.csv","w")
    for i in range(len(_testdata)):
        f.write(str(_result[i])+"\n")
    f.close()
