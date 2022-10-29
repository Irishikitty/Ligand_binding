#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:52:11 2022

@author: menghanlin
"""

import os
import pandas as pd
import pickle

metals = ('CU','FE','FE2','MG','NI','MN','K','NA','MO','CO','ZN','W','CA','CD','HG','V',
          'AU','BA','PB','PT','SM','SR','CU1')

temp = pd.read_pickle('./preprocesssing/dataset.pickle')

train = temp['train']
test = temp['test']
validation = temp['validation']

train = [i for i in train if i[5] in metals]
test = [i for i in test if i[5] in metals]
validation = [i for i in validation if i[5] in metals]


train_metal = [i[5] for i in train]
train_metal = pd.DataFrame(train_metal)
print(train_metal.value_counts())

test_metal = [i[5] for i in test]
test_metal = pd.DataFrame(test_metal)
print(test_metal.value_counts())

dict_a = {'train': train, 'test': test, 'validation':validation}
pickle.dump(dict_a, open('dataset_metal.pickle', 'wb'))

