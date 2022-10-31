import pandas as pd
import os
import numpy as np


train = pd.read_pickle('train_env_atoms_deque.pickle')
test = pd.read_pickle('test_env_atoms_deque.pickle')
validation = pd.read_pickle('validation_env_atoms_deque.pickle')

metals = ('CU','FE','FE2','MG','NI','MN','K','NA','MO','CO','ZN','W','CA','CD','HG','V',
                  'AU','BA','PB','PT','SM','SR','CU1')

train = [i for i in train if i[5] in metals]
test = [i for i in test if i[5] in metals]
validation = [i for i in validation if i[5] in metals]

train_metal = [i[5] for i in train]
train_metal = pd.DataFrame(train_metal)
train_stat = pd.DataFrame(train_metal.value_counts())

validation_metal = [i[5] for i in validation]
validation_metal = pd.DataFrame(validation_metal)
validation_stat = pd.DataFrame(validation_metal.value_counts())

test_metal = [i[5] for i in test]
test_metal = pd.DataFrame(test_metal)
test_stat = pd.DataFrame(test_metal.value_counts())
test_stat['metal'] = [i[0] for i in test_stat.index]
test_stat = test_stat.set_index('metal')

print(test_stat)
temp = train_stat.merge(validation_stat, how = 'outer', left_on = train_stat.index, right_on = validation_stat.index)
temp.key_0 = [i[0] for i in temp.key_0]
temp = temp.set_index('key_0')
temp = temp.merge(test_stat, how = 'outer', left_on = temp.index, right_on = test_stat.index)

temp.columns = ['metal','train','validation','test']
temp = temp.set_index('metal')
temp.loc[len(temp)] = temp.sum().values.tolist()
temp['metal'] = temp.index[:-1].values.tolist() + ['total']
temp = temp.set_index('metal')
temp = temp.fillna(0).astype('int32')

print(temp)

temp.to_csv('ligand_preprocessing_result.csv')



import pickle
# pickle.dump(train, open('train_metal_env.pickle','wb'))
# pickle.dump(validation, open('validation_metal_env.pickle','wb'))
# pickle.dump(test, open('test_metal_env.pickle','wb'))
