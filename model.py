# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 00:48:48 2021

@author: Rashmi Thekkath
"""

from pycaret.datasets import get_data
data = get_data("insurance")

from pycaret.regression import *
s = setup(data, target = 'charges', session_id = 123)

lr = create_model('lr')
plot_model(lr)
#linear regression - 10 fold cross validation

save_model(lr, model_name = 'C:/Users/rashm/Insurance_cloud/deployment_28042020')
#To save the linear regression model from this python notebook as a filein Insurance_cloud folder 
#Saved as a pkl file
