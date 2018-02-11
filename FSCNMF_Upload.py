# -*- coding: utf-8 -*-
#@author: Sambaran
"""
Created on Sun May 14 12:26:24 2017

@author: Sambaran Bandyopadhyay
"""

#%%

import time
import string
import sys
import os
import gc

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import csc_matrix as csc
from numpy import linalg as LA

#%%

gc.enable()

mainpath = os.path.dirname(os.path.realpath(sys.argv[0]))
filepath = mainpath



totalIter=10 #For example to run FSCNMF upto order 5, give 5 as input


adjacencyListFile = 'adjacencyList.csv'
contentFile='content.csv'
paperIdFile='paperId.csv'
fileLabels='labels.csv'



df_paperId = pd.read_csv(filepath+paperIdFile, header=None)


paperIdInd={}
ind=0
for item in df_paperId.ix[:,0]:
    paperIdInd[str(item)]=ind
    ind+=1

 
#%%       
#Processing the structure

print 'Processing structure started'

row=0
rowList=[]
columnList=[]
valueList=[]
with open(filepath+adjacencyListFile) as infile:
    for line in infile:
        reflists=line.split(',')
        reflists=map(string.rstrip,reflists) #Python's rstrip() method strips all kinds of trailing whitespace by default
        curr_item=reflists[0]
        reflists=reflists[1:]

        for item in reflists:
            if item in paperIdInd:
                rowList.append(paperIdInd[curr_item])
                columnList.append(paperIdInd[item])
                valueList.append(1)

        row+=1
        
#Create a sparse csc matrix
strucSpMat = csc((np.array(valueList), (np.array(rowList), np.array(columnList))), shape=(len(df_paperId), len(df_paperId)))

strucSpMat_org = strucSpMat


            
print 'Processing structure ended'

 #%%       
#Processing the content
    
print 'Processing content started'
    
C = pd.read_csv(filepath+contentFile, header=None)
C = C.as_matrix(columns=None)



true_labels = pd.read_csv(filepath+fileLabels, header=None)
true_labels = true_labels[0].tolist()

NoComm = len(set(true_labels))

#df_paperId = pd.read_csv(filepath+paperIdFile, header=None)


#%%

k=10*NoComm #Dimension of the embedding space


gc.collect()

#%%


small=pow(10,-3)

n=len(df_paperId)
d=C.shape[1]


accList=[]
accBest=0.0
opti1_values_best=[]
opti2_values_best=[]


outerIter=10 #15 #5 #20
opti1Iter=3
opti2Iter=3


runtime=[]

for noIterations in range(totalIter): #totalIter): #For example to run FSCNMF upto order 5, give 5 as input

    print 'FSCNMF++ of order {} has started'.format(noIterations+1)
    
    
    gc.collect()
    
    start_time = time.time()
    
    
    #For FSCNMF++
    
    strucSpMat = strucSpMat_org
        
    temp = strucSpMat_org
    for i in range(2, noIterations+2):
        temp = temp * strucSpMat_org
        strucSpMat = strucSpMat + temp
        
    strucSpMat = (strucSpMat).astype(float) / (noIterations+1)
    A = (strucSpMat.toarray()).astype(float)
    A = A.astype(float) #Get the matrix for FSCNMF of order (noIterations+1)
    
        
    A = np.nan_to_num(A)
    
    
    #Initializing matrices based on regular NMF
    NMFmodel1 = NMF(n_components=k, init='nndsvd', random_state=0)
    B1 = NMFmodel1.fit_transform(A)
    B2 = NMFmodel1.components_
    
    NMFmodel2 = NMF(n_components=k, init='nndsvd', random_state=0)
    U = NMFmodel2.fit_transform(C)
    V = NMFmodel2.components_


    
    B1=np.nan_to_num(B1) #Removing Nan or infinity, if any by numbers 0 or a large number respectively
    B2=np.nan_to_num(B2)
    U=np.nan_to_num(U)
    V=np.nan_to_num(V)
    
    
    B1_new=B1
    B2_new=B2
    U_new=U
    V_new=V
    
    
    const1 = np.ones((n,k))/k
    const2 = np.ones((k,d))/d
    const3 = np.ones((k,n))/k
    
    
    alpha1=1000 #10000.0 #1000 #match constraint
    alpha2=1.0 #0.001
    alpha3=1.0
    beta1=1000.0 #1000.0 #match constraint
    beta2=1.0
    beta3=1.0
    
    
    
    opti1_values=[]
    opti2_values=[]
    
    count_outer=1
    
    while True:
        
        print '\nOuter Loop {} started \n'.format(count_outer)
        
        #print '\nOptimization 1 starts \n'
        
        gamma = 0.001
        count1=1
        
        while True: #Optimization 1
            
        #    funcVal = 1.0/4.0 * pow( LA.norm(A-np.matmul(B,np.transpose(B)),'fro'), 2) + alpha1/2 * pow(LA.norm(B-U,'fro'),2) + alpha2 * np.abs(B).sum()
            funcVal = 1.0/2.0 * pow( LA.norm(A-np.matmul(B1,B2),'fro'), 2) + alpha1/2 * pow(LA.norm(B1-U,'fro'),2) + alpha2 * np.abs(B1).sum() + alpha3 * np.abs(B2).sum()
            
#                print funcVal
#                print 1.0/2.0 * pow( LA.norm(A-np.matmul(B1,B2),'fro'), 2)
#                print alpha1/2 * pow(LA.norm(B1-U,'fro'),2)
#                print alpha2 * np.abs(B1).sum() + alpha3 * np.abs(B2).sum()
#                print '\n'
            
            opti1_values.append(funcVal)
            

            B1_new = np.multiply(B1, np.divide( np.matmul(A,np.transpose(B2))+alpha1*U , (np.matmul(np.matmul(B1,B2), np.transpose(B2)) + alpha1*B1 + alpha2*B1).clip(min=small) ) ).clip(min=small) #Multiplicative update rule - aswin
            B2_new = np.multiply(B2, np.divide( np.matmul(np.transpose(B1),A), (np.matmul(np.transpose(B1),np.matmul(B1,B2))+beta3*B2).clip(min=small) )).clip(min=small)

                
            B1 = B1_new
            B2 = B2_new
            
            gamma = gamma/count1
            
            count1+=1
        
            if count1>opti1Iter:
                opti1_values.append(1.0/2.0 * pow( LA.norm(A-np.matmul(B1,B2),'fro'), 2) + alpha1/2 * pow(LA.norm(B1-U,'fro'),2) + alpha2 * np.abs(B1).sum() + alpha3 * np.abs(B2).sum())
                opti1_values.append(None)
                break
                
        
        count2=1
        gamma = 0.001
        while True:
            
            funcVal = 1.0/2.0 * pow(LA.norm(C - np.matmul(U,V), 'fro'), 2) + beta1/2 * pow(LA.norm(U-B1, 'fro'), 2) + beta2 * np.abs(U).sum() + beta3 * np.abs(V).sum()
            
#                print funcVal
#                print 1.0/2.0 * pow(LA.norm(C - np.matmul(U,V), 'fro'), 2)
#                print beta1/2 * pow(LA.norm(U-B1, 'fro'), 2)
#                print beta2 * np.abs(U).sum() + beta3 * np.abs(V).sum()
#                print '\n'
            
            opti2_values.append(funcVal)
            
            
            U_new = np.multiply(U, np.divide( np.matmul(C,np.transpose(V))+beta1*B1 , (np.matmul(np.matmul(U,V), np.transpose(V)) + beta1*U + beta2*U).clip(min=small) ) ).clip(min=small) #Multiplicative update rule - aswin
            V_new = np.multiply(V, np.divide( np.matmul(np.transpose(U),C), (np.matmul(np.transpose(U),np.matmul(U,V))+beta3*V).clip(min=small) )).clip(min=small)
            
            
            U = U_new
            V = V_new
            
            gamma = gamma/count2
            
            count2+=1
        
            if(count2>opti2Iter):
                opti2_values.append(1.0/2.0 * pow(LA.norm(C - np.matmul(U,V), 'fro'), 2) + beta1/2 * pow(LA.norm(U-B1, 'fro'), 2) + beta2 * np.abs(U).sum() + beta3 * np.abs(V).sum())
                opti2_values.append(None)
                break
            
            
        
        count_outer+=1
        if(count_outer>outerIter):
            break
        
    
    
    #Writing the opti values in the files for this order of FSCNMF++
    
    optiFile = open(filepath+'FSCNMF++_{}_OptiValues.csv'.format(noIterations+1), 'w')

    for item in opti1_values:
        optiFile.write(str(item)+'\t')
    
    optiFile.write('\n')
    
    for item in opti2_values:
        optiFile.write(str(item)+'\t')
    
    optiFile.close()
    
          
        
    B1 = (B1_new/((np.abs(B1_new).sum(axis=1)).reshape(n,1))).clip(min=small) #Normalising the rows
    B2 = (B2_new/((np.abs(B2_new).sum(axis=1)).reshape(k,1))).clip(min=small) #Normalising the rows
    
    U = (U_new/((np.abs(U_new).sum(axis=1)).reshape(n,1))).clip(min=small)
    V = (V_new/((np.abs(V_new).sum(axis=1)).reshape(k,1))).clip(min=small)
    
    
    
    np.savetxt(filepath+'FSCNMF++_order{}_rep.txt'.format(noIterations+1), B1)
    
    gc.collect()
    
    print 'Embedding done for FSCNMF++ of oreder {}'.format(noIterations+1)
    
    
