# -*- coding: utf-8 -*-

import  numpy as np
import pandas as pd
import scipy.io as sc

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
dd = np.zeros((1,12))
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
    cc=pd.DataFrame(c)
    cc[12]=m
    dd=dd.append(cc,ignore_index=True)
    m=m+1
    i=i+9
dd = dd.drop(0, axis=0)



from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix  


x=dd.iloc[:,:12]
y=dd.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)


'''LinearDiscriminantAnalysis'''
################################################################################
LDA = LinearDiscriminantAnalysis();
LDA.fit(X_train, y_train)
print('test accuracy lda-',LDA.score(X_test, y_test,sample_weight= None))
y_pred = LDA.predict(X_test)

from sklearn.svm import SVC

'''Support vector machine linear'''
################################################################################
'''svcx=SVC(kernel='linear')
svcx.fit(X_train,y_train)
print('test accuracy svm:',svcx.score(X_test,y_test))
y_pred =svcx.predict(X_test)
'''
'''QuadraticDiscriminantAnalysis'''
################################################################################
'''QDA = QuadraticDiscriminantAnalysis();
QDA.fit(X_train, y_train)
print('test qda:',QDA.score(X_test, y_test,sample_weight= None))
y_pred = QDA.predict(X_test)
'''

