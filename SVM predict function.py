import numpy as np


def rbfdot(x1,x2,gamma):
    return np.exp(-gamma*np.sum((x1-x2)**2,axis=1))

def predict(Xt,model):
    coef=model.dual_coef_[0]
    vec=model.support_vectors_
    gamma=model.get_params()["gamma"]
    #print (model.dual_coef_)
    cnt=0
    """b=0
    for i in vec:
        val=np.sum(coef*rbfdot(vec,i,gamma))
        b=b+val"""
    b=model.intercept_[0]#b/len(vec)
    pred=np.zeros(len(Xt))
    cnt=0
    for i in Xt:
        val=np.sum(coef*rbfdot(vec,i,gamma))+b
        #print(val)
        if val>=0:
            pred[cnt]=1
        else:
            pred[cnt]=0
        cnt+=1
    return pred
