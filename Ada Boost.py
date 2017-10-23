import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

def base_learn(X,Y,weight):
    lr=LogisticRegression()
    lr.fit(X,Y,sample_weight=weight)
    return lr


class adaboost:
    def __init__(self,base_learn,early_stop=False):
        self.blearn=base_learn
        self.es=early_stop
        pass
    def train(self,X,Y,num):
        assert(X.shape[0]==Y.shape[0])
        self.ll=num
        self.learner=[None]*num
        self.lw=[None]*num
        l=X.shape[0]
        ins_weight=np.zeros([l])
        ins_weight+=float(1)/l
        for i in range(num):
            self.learner[i]=self.blearn(X,Y,ins_weight)
            yp=self.learner[i].predict(X)
            score=np.sum(ins_weight*(yp==Y))
            #print(score)
            if score<0.5 and self.es:
                self.ll=i+1
                break
            self.lw[i]=np.log(score/(1-score))/2
            ins_weight=ins_weight*np.exp(-self.lw[i]*yp*Y)
            ins_weight/=np.sum(ins_weight)
    def predict(self,X):
        l=X.shape[0]
        rst=np.zeros([l])
        for i in  range(self.ll):
            rst+=self.lw[i]*self.learner[i].predict(X)
        rst[rst>0]=1
        rst[rst<=0]=0
        return rst
            
            
def kfold(X,Y,fold_num,ada_param):
    kf = KFold( n_splits=fold_num)
    cnt=0
    for train_index,test_index in kf.split(X):
        cnt+=1
        Xtrain=X[train_index]
        Ytrain=Y[train_index]
        Xtest=X[test_index]
        Ytest=Y[test_index]
        adb=adaboost(base_learn=base_learn)
        adb.train(Xtrain,Ytrain,ada_param)
        Ypred=adb.predict(Xtest)
        output=np.concatenate(((test_index+1).reshape(-1,1),Ypred.reshape(-1,1)),axis=1)
        #print(output)
        np.savetxt(fname='experiments/base%d_fold%d.csv'%(ada_param,cnt),X=output,delimiter=',',fmt='%d')
            
if __name__=="__main__":
    X = np.genfromtxt(fname = 'data.csv',delimiter = ',')
    Y = np.genfromtxt(fname = 'targets.csv',delimiter = ',')
    Y[Y==0]=-1
    for i in [1,5,10,100]:
        kfold(X,Y,10,i)
