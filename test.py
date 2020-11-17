# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:14:32 2019

@author: Oddysseus
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sc


def stat(x):
    mean=np.mean(x)
    std=np.std(x)
    rms=np.mean(x**2)**(0.5)
    
    return mean,std,rms

def feat_1(x,win_size,wininc):
    df = pd.DataFrame(x)
    
    datasize =len(df)
    nsignals = len(df.columns)
    st = 0
    en = win_size  
    
    numwin=int(((datasize-win_size)/wininc)+1)
    fed = np.zeros((int(numwin),(nsignals)*3))
    for i in range(0,numwin):
        curwin=df.iloc[st:en,:]
        kt=stat(curwin)
        
        k=0
        while k!=nsignals*3:
            
        #fet = np.empty(3, dtype=object)
            for j in range(0,nsignals):
                fed[i][k]=kt[0][j]
                fed[i][k+1]=kt[1][j]
                fed[i][k+2]=kt[2][j]
                k=k+3
        st = st+wininc
        en = en+wininc
    return fed
l=['A6_Ball_high_t1.mat','A6_Ball_high_t2.mat','A6_Ball_high_t3.mat',
   'A6_Ball_low_t1.mat','A6_Ball_low_t2.mat','A6_Ball_low_t3.mat',
   'A6_Ball_med_t1.mat','A6_Ball_med_t2.mat','A6_Ball_med_t3.mat',
   'A6_Ind_high_t1.mat','A6_Ind_high_t2.mat','A6_Ind_high_t3.mat',
   'A6_Ind_low_t1.mat','A6_Ind_low_t2.mat','A6_Ind_low_t3.mat',
   'A6_Ind_med_t1.mat','A6_Ind_med_t2.mat','A6_Ind_med_t3.mat',
   'A6_LRMI_high_t1.mat','A6_LRMI_high_t2.mat','A6_LRMI_high_t3.mat',
   'A6_LRMI_low_t1.mat','A6_LRMI_low_t2.mat','A6_LRMI_low_t3.mat',
   'A6_LRMI_med_t1.mat','A6_LRMI_med_t2.mat','A6_LRMI_med_t3.mat',
   'A6_Th_high_t1.mat','A6_Th_high_t2.mat','A6_Th_high_t3.mat',
   'A6_Th_low_t1.mat','A6_Th_low_t2.mat','A6_Th_low_t3.mat',
   'A6_Th_med_t1.mat','A6_Th_med_t2.mat','A6_Th_med_t3.mat',
   'A6_ThInd_high_t1.mat','A6_ThInd_high_t2.mat','A6_ThInd_high_t3.mat',
   'A6_ThInd_low_t1.mat','A6_ThInd_low_t2.mat','A6_ThInd_low_t3.mat',
   'A6_ThInd_med_t1.mat','A6_ThInd_med_t2.mat','A6_ThInd_med_t3.mat',
   'A6_ThIndMid_high_t1.mat','A6_ThIndMid_high_t2.mat','A6_ThIndMid_high_t3.mat',
   'A6_ThIndMid_low_t1.mat','A6_ThIndMid_low_t2.mat','A6_ThIndMid_low_t3.mat',
   'A6_ThIndMid_med_t1.mat','A6_ThIndMid_med_t2.mat','A6_ThIndMid_med_t3.mat']
i=0
m=1
dd = np.zeros((1,36))
dd=pd.DataFrame(dd)
while i<54:
    df1=(sc.loadmat(l[i])['t1'])
    df2=(sc.loadmat(l[i+1])['t2'])
    df3=(sc.loadmat(l[i+2])['t3'])
    df4=(sc.loadmat(l[i+3])['t1'])
    df5=(sc.loadmat(l[i+4])['t2'])
    df6=(sc.loadmat(l[i+5])['t3'])
    df7=(sc.loadmat(l[i+6])['t1'])
    df8=(sc.loadmat(l[i+7])['t2'])
    df9=(sc.loadmat(l[i+8])['t3'])
    c=np.vstack((df1,df2,df3,df4,df5,df6,df7,df8,df9))
    cc=feat_1(c,1000,150)
    cc=pd.DataFrame(cc)
    cc[36]=m
    dd=dd.append(cc,ignore_index=True)  
    m=m+1
    i=i+9
dd = dd.drop(0, axis=0)


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


x=dd.iloc[:,:36]
y=dd.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)


'''LinearDiscriminantAnalysis'''
################################################################################
LDA = LinearDiscriminantAnalysis();
LDA.fit(X_train, y_train)
y_pred = LDA.predict(X_test)
print(LDA.score(X_train, y_train,sample_weight= None))
print('test-',LDA.score(X_test, y_test,sample_weight= None))


from sklearn.svm import SVC

'''Support vector machine linear'''
################################################################################
'''svcx=SVC(kernel='linear')
svcx.fit(X_train,y_train)
print(svcx.score(X_train,y_train))
print('test:',svcx.score(X_test,y_test))
'''

'''QuadraticDiscriminantAnalysis'''
################################################################################
'''QDA = QuadraticDiscriminantAnalysis();
QDA.fit(X_train, y_train)
y_pred = QDA.predict(X_test)
print(QDA.score(X_train, y_train,sample_weight= None))
print('test:',QDA.score(X_train, y_train,sample_weight= None))
'''
