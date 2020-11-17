# -*- coding: utf-8 -*-
"""
Created on  Jul  8 1:45 PM 2019

@author: mss2015
"""
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.io as sc


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):    
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Reds')

    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=60,fontsize=14)
        plt.yticks(tick_marks, target_names,fontsize=14)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),fontsize=14,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),fontsize=12,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True Class',fontsize=14)
    plt.xlabel('Predicted Class',fontsize=14)
    plt.xlabel('Predicted Class\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass),fontsize=14)
    plt.show()







def stat(x):
    u0 = (np.sum((x**2),axis=0))**(0.5)
    u2 = (np.sum((np.diff(x,n=2,axis=0)**2),axis=0))**(0.5)
    u4 = (np.sum((np.diff(x,n=4,axis=0)**2),axis=0))**(0.5)
    NP = u4/u2
    ZC = u2/u0
    f1 = u0/NP
    f2 = u0/ZC
    f3 = u0 - u2
    return (np.concatenate((f1,f2,f3,u2), axis=None))

def feat_1(x,win_size,wininc):
    df = pd.DataFrame(x)
    
    datasize =len(df)
    nsignals = len(df.columns)
    st = 0
    en = win_size  
    
    numwin =((datasize-win_size)/wininc)+1
    fed = np.zeros((int(numwin),(nsignals)*4))
    for i in range(0,int(numwin)):
        curwin= x[st:en]
        kt = stat(curwin)
        fed[i,:] = kt
        st = st+wininc
        en = en+wininc
    return fed
l=[ 'A6_Th_high_t1.mat','A6_Th_high_t2.mat','A6_Th_high_t3.mat',
   'A6_Th_low_t1.mat','A6_Th_low_t2.mat','A6_Th_low_t3.mat',
   'A6_Th_med_t1.mat','A6_Th_med_t2.mat','A6_Th_med_t3.mat',
   'A6_Ind_high_t1.mat','A6_Ind_high_t2.mat','A6_Ind_high_t3.mat',
   'A6_Ind_low_t1.mat','A6_Ind_low_t2.mat','A6_Ind_low_t3.mat',
   'A6_Ind_med_t1.mat','A6_Ind_med_t2.mat','A6_Ind_med_t3.mat',   
   'A6_ThInd_high_t1.mat','A6_ThInd_high_t2.mat','A6_ThInd_high_t3.mat',
   'A6_ThInd_low_t1.mat','A6_ThInd_low_t2.mat','A6_ThInd_low_t3.mat',
   'A6_ThInd_med_t1.mat','A6_ThInd_med_t2.mat','A6_ThInd_med_t3.mat',
   'A6_ThIndMid_high_t1.mat','A6_ThIndMid_high_t2.mat','A6_ThIndMid_high_t3.mat',
   'A6_ThIndMid_low_t1.mat','A6_ThIndMid_low_t2.mat','A6_ThIndMid_low_t3.mat',
   'A6_ThIndMid_med_t1.mat','A6_ThIndMid_med_t2.mat','A6_ThIndMid_med_t3.mat',
   'A6_LRMI_high_t1.mat','A6_LRMI_high_t2.mat','A6_LRMI_high_t3.mat',
   'A6_LRMI_low_t1.mat','A6_LRMI_low_t2.mat','A6_LRMI_low_t3.mat',
   'A6_LRMI_med_t1.mat','A6_LRMI_med_t2.mat','A6_LRMI_med_t3.mat',   
   'A6_Ball_high_t1.mat','A6_Ball_high_t2.mat','A6_Ball_high_t3.mat',
   'A6_Ball_low_t1.mat','A6_Ball_low_t2.mat','A6_Ball_low_t3.mat',
   'A6_Ball_med_t1.mat','A6_Ball_med_t2.mat','A6_Ball_med_t3.mat']
i=0
m=1
dd = np.zeros((1,48))
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
    cc=feat_1(c,400,150)
    cc=pd.DataFrame(cc)
    cc[48]=m
    dd=dd.append(cc,ignore_index=True)
    m=m+1
    i=i+9
dd = dd.drop(0, axis=0)



from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix  


x=dd.iloc[:,:48]
y=dd.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)


'''LinearDiscriminantAnalysis'''
################################################################################
'''LDA = LinearDiscriminantAnalysis();
LDA.fit(X_train, y_train)
print('test accuracy-',LDA.score(X_test, y_test,sample_weight= None))
y_pred = LDA.predict(X_test)
array=confusion_matrix(y_test,y_pred)
'''

from sklearn.svm import SVC

'''Support vector machine linear'''
################################################################################
'''svcx=SVC(kernel='linear')
svcx.fit(X_train,y_train)
print('test accuracy:',svcx.score(X_test,y_test))
y_pred =svcx.predict(X_test)
array=confusion_matrix(y_test,y_pred)
'''

'''QuadraticDiscriminantAnalysis'''
################################################################################
QDA = QuadraticDiscriminantAnalysis();
QDA.fit(X_train, y_train)
print('test:',QDA.score(X_test, y_test,sample_weight= None))
y_pred = QDA.predict(X_test)
array=confusion_matrix(y_test,y_pred)
'''
plot_confusion_matrix(array,
                      target_names = ['A1', 'A2', 'A3','A4', 'A5', 'A6'],
                      normalize    = True,)
'''


A9sample=   ( sc.loadmat("A9_Ball_high_t2.mat")['t2'])


