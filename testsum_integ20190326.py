# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:38:18 2019

@author: LYY
"""

import numpy as np
import bilby
import matplotlib.pyplot as plt
import scipy.integrate as integrate
#from bilby.core.likelihood import GaussianLikelihood
from bilby.core.prior import Uniform
from bilby.core.sampler import run_sampler
#import deepdish
#import seaborn as sns
from random import sample 
from bilby.hyper.likelihood import HyperparameterLikelihood
import pandas as pd
def EVx(x,a):
    return x**a

def pxinteg(x,a):
    return EVx(x,a)/(integrate.quad(EVx,1,100,args=(a))[0])

xa=np.zeros(100)
for i in range(100):
    xa[i]=i*1+1
def sEVx(a):
    return np.sum(EVx(xa,a))
p3=pxinteg(xa,2)
#################################################
dataDLM1M2=xa
Num=np.zeros(len(dataDLM1M2))
Ev=np.zeros(len(dataDLM1M2))
p3=np.zeros(len(dataDLM1M2))
E=np.zeros(len(p3))   
for i in range(len(p3)):
    E[i]=p3[i]*10**5
    E[i]=int(E[i])
emp=[i for i in range(len(E)) if E[i] == 0]
if len(emp) >= 1:
    num_0=len(emp)
    no0DLM1M2=np.zeros(len(dataDLM1M2)-num_0)
    no0E=np.zeros(len(E)-num_0)
    no0DLM1M2=np.delete(dataDLM1M2,emp,0)
    no0E=np.delete(E,emp)
else:
    no0E=E
    no0DLM1M2=dataDLM1M2

Nevent=np.zeros(int(np.sum(no0E)))
sa=0
for i in range(len(no0E)):
    Nevent[int(sa):int(sa+no0E[i])]= np.repeat(i,int(no0E[i]))
    sa=no0E[i]+sa
choice_size=100
JB=np.zeros(choice_size)#####jiaobiao
JB=sample(list(Nevent),choice_size)
cos=list()
for i in range(len(JB)):
    kk=int(JB[i])
    co=no0DLM1M2[kk]  
    cos.append(co)
df=pd.DataFrame(cos)
print(cos)
np.savetxt('3Ddata2Dpara20190326part1injection.txt',df.values,fmt='%.3f')