# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:44:10 2019

@author: Oddysseus
"""

# -*- coding: utf-8 -*-
"""
Created on  Jul  8 1:45 PM 2019

@author: mss2015
"""
import pandas as pd
import numpy as np
from numpy import newaxis
import itertools
import matplotlib.pyplot as plt
import scipy.io as sc
          






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






X_train = X_train.rename_axis('ID').values
X_train=X_train[:,:,newaxis,newaxis]
print(X_train.shape)




from tensorflow.keras.models import Sequential



#from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Activation
#create model
model = Sequential()
#add model layers

model.add(Conv2D(64, kernel_size=3,  input_shape=(48,1,1)))
model.add(Activation('relu'))

model.add(Conv2D(32, kernel_size=3))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)


"""


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pickle
'''
pickle_in = open("X.pickle","rb")
x = pickle.load(pickle_in)
X=np.divide(x,255.0)
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)'''
model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=(1,50,50)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
import tensorflow as tf
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x, y, batch_size=32, epochs=3, validation_split=0.3)
"""