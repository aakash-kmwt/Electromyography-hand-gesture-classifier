# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:56:21 2019

@author: Oddysseus
"""

import numpy as num
import pandas as pd
import scipy.io as sc 

file_names =['A6_Ball_high_t1.mat','A6_Ball_high_t2.mat','A6_Ball_high_t3.mat',
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

df1=(sc.loadmat(file_names[0])['t1'])

print(df1)