import math
import pickle
import numpy as np

class BeyesianClassifier:
    def __init__(self,feature_num,label_num,feature_type=None,dis_fea_range=None,eps=1e-5):
        assert(feature_type is not None)
        self.fn=feature_num
        self.ft=feature_type
        self.ln=label_num
        self.eps=eps
        self.dfr=dis_fea_range
        self.param=[None for i in range(self.fn)]
        self.param2=[None for i in range(self.ln)]
        self.cntalpha=0.5

class NaiveBeyesianClassifier(BeyesianClassifier):

    def __init__(self,feature_num,label_num,feature_type=None,dis_fea_range=None,eps=1e-5):
        super(NaiveBeyesianClassifier,self).__init__(feature_num,label_num,feature_type=feature_type,dis_fea_range=dis_fea_range,eps=eps)

    def prob(self,x,c):
        cnt=0
        for i in range(self.fn):
            if self.ft[i]==0:
                tp=int(x[i])
                pxgc=self.param[i][tp][c]
            else:
                if self.param[i][c*3]>=2:
                    pxgc=-math.pow((x[i]-self.param[i][c*3+1])/(2*self.param[i][c*3+2]),2)-math.log(self.param[i][c*3+2])
                else:
                    pxgc=-1/self.eps
            cnt+=pxgc
        cnt+=self.param2[c]
        return cnt
                
    def preprocess(self,data):
        var=data

    def train(self,data,label):
        assert(len(label)==len(data))
        l=len(data)
        self.totnum=l

        for j in range(self.fn):
            if self.ft[j]==0:
                self.param[j]=[[self.cntalpha for k in range(self.ln)] for m in range(self.dfr[j])]
            else:
                self.param[j]=[0]*(3*self.ln)

        self.param2=[self.cntalpha for i in range(self.ln)]

        for i in range(l):
            assert (label[i] in list(range(self.ln)))
            for j in range(self.fn):
                if self.ft[j]==0: # discrete 
                    _tp=int(data[i][j])
                    assert(_tp in list(range(self.dfr[i])))
                    self.param[j][_tp][label[i]]+=1
                else:
                    _base=label[i]*3
                    self.param[j][_base+0]+=1
                    self.param[j][_base+1]+=data[i][j]
                    self.param[j][_base+2]+=data[i][j]*data[i][j]
            self.param2[label[i]]+=1            

        for i in range(self.fn):
            if self.ft[i]==0:
                for j in range(self.ln):
                    _cnt=0
                    for k in range(self.dfr[i]):
                        _cnt+=self.param[i][k][j]
                    for k in range(self.dfr[i]):
                        self.param[i][k][j]=math.log(float(self.param[i][k][j])/_cnt)
            else:
                for j in range(self.ln):
                    _cnt=self.param[i][j*3]
                    if _cnt>=2:
                        _e=float(self.param[i][j*3+1])
                        _e2=float(self.param[i][j*3+2])
                        self.param[i][j*3+1]=_e/_cnt
                        self.param[i][j*3+2]=math.sqrt((_e2-_e*_e/_cnt)/(_cnt-1))+self.eps

        for i in range(self.ln):
            self.param2[i]=math.log(float(self.param2[i])/self.totnum)
    def predict(self,data):
        l=len(data)
        label=[None]*l
        for i in range(l):
            _tpl=None
            _ch=0
            for j in range(self.ln):
                pc=self.prob(data[i],j)
                if _tpl is None or _tpl<pc:
                    _tpl=pc
                    _ch=j
            label[i]=_ch
        return label

if __name__=="__main__":
    X=pickle.load(open('train_data.pkl','rb')).todense() # unsupported in Python 2
    y=pickle.load(open('train_targets.pkl','rb'))
    Xt=pickle.load(open('test_data.pkl','rb')).todense()
    X=X.tolist()
    y=y.tolist()
    Xt=Xt.tolist()

    #print(X)

    ft=[0]*2500
    ft.extend([1]*2500)
    dfr=[2]*5000
    ln=5
    fn=5000
    NB=NaiveBeyesianClassifier(fn,ln,ft,dfr)
    NB.train(X,y)
    Yt=NB.predict(Xt)
    f=open("test_predictions.csv","w")
    for i in range(len(Yt)):
        f.write(str(Yt[i])+"\n")
    f.close()
