#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 21:40:44 2022

@author: menghanlin
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression

os.listdir('.')

starting = pd.read_pickle('starting_rmsd.pickle')
rmsd_values = pd.read_pickle('ligand_rmsd.pickle')
pred_true = pd.read_pickle('pred_true.pickle')
ligandlen = pd.read_pickle('ligandlen.pickle')

temp = pred_true[0][0]
tmp = np.stack(pred_true[0][1])
temp[:,1:]
fig = plt.figure()
plt.figure(figsize=(8, 6), dpi=80)
ax = plt.axes(projection='3d')
ax.scatter3D(temp[:,1],temp[:,2],temp[:,3],c='blue', cmap='viridis')
ax.scatter3D(tmp[:,1],tmp[:,2],tmp[:,3], c='red')
plt.show()

# accu/ligandlen
model = LinearRegression().fit(np.array(ligandlen).reshape(-1,1), rmsd_values)
model.coef_
model.intercept_

plt.scatter(ligandlen, rmsd_values)
x = np.linspace(0, 49, 1000)
plt.plot(x, x*model.coef_+model.intercept_, '-.', c = 'orange')
plt.xlabel('ligand length')
plt.ylabel('RMSD')
plt.title('neutral charge ligand')

np.mean(rmsd_values), np.median(rmsd_values), np.std(rmsd_values)


for i in range(len(pred_true)):
    pred, true = pred_true[i]


pos = pd.read_pickle('df_train_pos.pkl')
pos_test = pd.read_pickle('df_test_neu.pkl')

ligands = [i[1] for i in pos]
Counter(ligands)
ligandlens = [len(i[2]) for i in pos] + [len(i[2]) for i in pos_test]
pd.Series([len(i[2]) for i in pos]).describe()

model.coef_


