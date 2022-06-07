import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import expon
import math
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import random

def generate_data():
  #generating pt (target)
  #. normal distribution with mean 500 and std 
  size = 50000
  pt_tar = norm.rvs(500, 50, size = size)
  label_tar = list()
  for pt in pt_tar:
    label_tar.append('target')
  pt_tar_labelled = zip(pt_tar, label_tar)

  #generating pt (source)
  #. truncated normal distribution with mean 0, std 100, truncated between 0 and 1000
  clip_a, clip_b, mean, std = 0, 1000, 0, 100
  a, b = (clip_a - mean) / std, (clip_b - mean) / std
  pt_src = truncnorm.rvs(a, b, loc = mean, scale = std, size = size)
  label_src = list()
  for pt in pt_src:
    label_src.append('source')
  pt_src_labelled = zip(pt_src, label_src)

  #generating pt (background)
  #. exponential distribution with loc 0 and scale 200
  pt_back = expon.rvs(0, 200, size = size)
  label_back = list()
  for pt in pt_back:
    label_back.append('background')
  pt_back_labelled = zip(pt_back, label_back)

  pt_list = list(pt_tar_labelled) + list(pt_src_labelled) + list(pt_back_labelled)

  #construct dataframe with pt values
  df = pd.DataFrame(pt_list, columns=['pt', 'label'])

  #one-hot encode labels
  df = pd.get_dummies(df)

  #generate features
  f1_list = list()
  f2_list = list()
  for i in range(len(df)):
    pt = df.iloc[i, df.columns.get_loc('pt')]
    if (df.iloc[i,df.columns.get_loc('label_background')] == 1):
      l = -1
    else:
      l = 1
    theta = norm.rvs(0, math.pi/3)
    f1 = pt**0.2+l*pt*(math.sin(theta)-0.5)+50
    f1_list.append(f1)
    theta = norm.rvs(0, math.pi/3)
    f2 = f2 = pt**1.1+l*pt*(math.cos(theta)-0.5)+5
    f2_list.append(f2)

  df.insert(1, 'f1', f1_list)
  df.insert(2, 'f2', f2_list)

  return df

train_df = generate_data()

df_src = train_df.loc[(train_df['label_source'] == 1)]
df_back = train_df.loc[(train_df['label_background'] == 1)]
df_tar = train_df.loc[(train_df['label_target'] == 1)]

#plot pt

bins = np.linspace(0, 1500, 100)
n_src, bins_src, patches_src = plt.hist(df_src['pt'].to_numpy(), bins, density=True, histtype='step', label = 'Source')
n_tar, bins_tar, patches_tar = plt.hist(df_tar['pt'].to_numpy(), bins, density=True, histtype='step', label = 'Target')
n_back, bins_back, patches_back = plt.hist(df_back['pt'].to_numpy(), bins, density=True, histtype='step', label = 'Background')
plt.legend(loc='best')
plt.show()

#reweight pt Source to pt Background MANUAL

#compute bin weights
weights = list()
for i in range(0, 99):
  weights.append(n_back[i]/n_src[i])

#assign weights for each pt value
train_df['weight'] = 1.0 #default weight
for i in range(train_df.shape[0]):
  if train_df.iloc[i, train_df.columns.get_loc('label_source')] == 1:
    for j in range(0, len(bins_src)):
      if (bins_src[j] < train_df.iloc[i, 0] <= bins_src[j+1]): #is src and is in bin size
        train_df.iloc[i, train_df.columns.get_loc('weight')] = weights[j]
        break


sum_src = 0
for i in range(len(df_src)):
  sum_src = sum_src + df_src.iloc[i,6]

class_weights = {
    0: 50000/sum_src,
    1: 1
}

df_src = train_df.loc[(train_df['label_source'] == 1)]
df_back = train_df.loc[(train_df['label_background'] == 1)]
df_tar = train_df.loc[(train_df['label_target'] == 1)]

#plot reweighted pt

bins = np.linspace(0, 1500, 100)
plt.hist(df_src['pt'].to_numpy(), bins, weights = df_src['weight'].to_numpy(), density=True, histtype='step', label = 'Source (Reweighted)')
#plt.hist(df_src['pt'].to_numpy(), bins, weights = weights_model, density=True, histtype='step', label = 'Source (Reweighted)')
plt.hist(df_tar['pt'].to_numpy(), bins, density=True, histtype='step', label = 'Target')
plt.hist(df_back['pt'].to_numpy(), bins, density=True, histtype='step', label = 'Background')
plt.legend(loc='best')
plt.show()

#plot f1 unweighted

bins = np.linspace(-1000, 500, 50)
plt.hist(df_src['f1'].to_numpy(), bins, density=True, histtype='step', label = 'Source')
plt.hist(df_tar['f1'].to_numpy(), bins, density=True, histtype='step', label = 'Target')
plt.hist(df_back['f1'].to_numpy(), bins, density=True, histtype='step', label = 'Background')
plt.legend(loc='best')
plt.show()

#plot f1 reweighted

bins = np.linspace(-1000, 500, 50)
plt.hist(df_src['f1'].to_numpy(), bins, weights = df_src['weight'].to_numpy(), density=True, histtype='step', label = 'Source (Reweighted)')
plt.hist(df_tar['f1'].to_numpy(), bins, density=True, histtype='step', label = 'Target')
plt.hist(df_back['f1'].to_numpy(), bins, density=True, histtype='step', label = 'Background')
plt.legend(loc='upper left')
plt.show()

#plot f2 unweighted

bins = np.linspace(0, 1500, 50)
plt.hist(df_src['f2'].to_numpy(), bins, density=True, histtype='step', label = 'Source')
plt.hist(df_tar['f2'].to_numpy(), bins, density=True, histtype='step', label = 'Target')
plt.hist(df_back['f2'].to_numpy(), bins, density=True, histtype='step', label = 'Background')
plt.legend(loc='best')
plt.show()

#plot f2 reweighted

bins = np.linspace(0, 1500, 50)
plt.hist(df_src['f2'].to_numpy(), bins, weights = df_src['weight'], density=True, histtype='step', label = 'Source (Reweighted)')
plt.hist(df_tar['f2'].to_numpy(), bins, weights = df_tar['weight'], density=True, histtype='step', label = 'Target')
plt.hist(df_back['f2'].to_numpy(), bins, weights = df_back['weight'], density=True, histtype='step', label = 'Background')
plt.legend(loc='best')
plt.show()

train_df = shuffle(train_df)
train_df = train_df.loc[(train_df['label_source'] == 1) | (train_df['label_background'] == 1)]

#UU
#output 1 means background

scaler = StandardScaler()
X = scaler.fit_transform(train_df[['f1','f2']].to_numpy().reshape(-1,2))
y = train_df[['label_background']].to_numpy()

model = keras.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape = (2,)))
model.add(layers.Dense(8,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=.001),loss='mean_squared_error',metrics=['accuracy'])
model.fit(x=X,y=y,validation_split=.1,batch_size=16,epochs=10,shuffle=True,verbose=1)
model.save('unparameterized_unweighted_src')

#PU
#output 1 means background

scaler = StandardScaler()
X = scaler.fit_transform(train_df[['f1','f2','pt']].to_numpy().reshape(-1,3))
y - train_df[['label_background']].to_numpy()

model = keras.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape = (3,)))
model.add(layers.Dense(8,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=.001),loss='mean_squared_error',metrics=['accuracy'])
model.fit(x=X,y=y,validation_split=.1,batch_size=16,epochs=10,shuffle=True,verbose=1)
model.save('parameterized_unweighted_src')

#UR
#output 1 means background

scaler = StandardScaler()
X = scaler.fit_transform(train_df[['f1','f2']].to_numpy().reshape(-1,2))
y = train_df[['label_background']].to_numpy()

model = keras.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape = (2,)))
model.add(layers.Dense(8,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=.001),loss='mean_squared_error',metrics=['accuracy'])
model.fit(x=X,y=y,validation_split=.1,batch_size=16,epochs=10,shuffle=True,verbose=1,sample_weight=train_df['weight'].to_numpy(),class_weight=class_weights)
model.save('unparameterized_reweighted_src')

#PR
#output 1 means background

scaler = StandardScaler()
X = scaler.fit_transform(train_df[['f1','f2','pt']].to_numpy().reshape(-1,3))
y - train_df[['label_background']].to_numpy()

model = keras.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape = (3,)))
model.add(layers.Dense(8,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=.001),loss='mean_squared_error',metrics=['accuracy'])
model.fit(x=X,y=y,validation_split=.1,batch_size=16,epochs=10,shuffle=True,verbose=1,sample_weight=train_df['weight'].to_numpy(),class_weight=class_weights)
model.save('parameterized_reweighted_src')

def plot_cfm(labels, pred, title):
  cf_matrix = confusion_matrix(labels, pred)
  ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
  ax.set_title(title + '\nf1 score = ' + str(f1_score(labels,pred)))
  ax.set_xlabel('Predicted Values')
  ax.set_ylabel('Actual Values ')
  ax.xaxis.set_ticklabels(['False','True'])
  ax.yaxis.set_ticklabels(['False','True'])
  plt.show()

def make_bin(y_pred):
  y_pred_bin = list()
  for pred in y_pred:
    if pred[0] > .5:
      y_pred_bin.append(1)
    else:
      y_pred_bin.append(0)
  return y_pred_bin
  
 #evaluating models

test_df = generate_data()

u_models = list()
p_models = list()
u_u_model = keras.models.load_model('unparameterized_unweighted_src')
u_models.append((u_u_model, 'Unparameterized-Unweighted'))
p_u_model = keras.models.load_model('parameterized_unweighted_src')
p_models.append((p_u_model, 'Parameterized-Unweighted'))
u_w_model = keras.models.load_model('unparameterized_reweighted_src')
u_models.append((u_w_model, 'Unparameterized-Reweighted'))
p_w_model = keras.models.load_model('parameterized_reweighted_src')
p_models.append((p_w_model, 'Parameterized-Reweighted'))

test_df = test_df.loc[(test_df['label_source'] == 1) | (test_df['label_background'] == 1)]

scaler = StandardScaler()
X = scaler.fit_transform(test_df[['f1', 'f2']].to_numpy().reshape(-1,2))
y = test_df['label_background'].to_numpy()

count = 1

for model in u_models:
  y_pred = model[0].predict(X)
  #y_pred_uu = list()
  if count == 1:
    y_pred_uu = np.array(make_bin(y_pred))
  if count == 2:
    y_pred_uw = np.array(make_bin(y_pred))
  #plot_cfm(y,make_bin(y_pred),model[1])
  fpr, tpr, thresholds = roc_curve(y, y_pred)
  auc = roc_auc_score(y, y_pred)
  plt.plot(fpr, tpr, label=model[1] + ' - AUC= ' + str(auc))
  count = count + 1

X = scaler.fit_transform(test_df[['f1', 'f2','pt']].to_numpy().reshape(-1,3))

for model in p_models:
  y_pred = model[0].predict(X)
  if count == 3:
    y_pred_pu = np.array(make_bin(y_pred))
  if count == 4:
    y_pred_pw = np.array(make_bin(y_pred))
  #plot_cfm(y,make_bin(y_pred),model[1])
  fpr, tpr, thresholds = roc_curve(y, y_pred)
  auc = roc_auc_score(y, y_pred)
  plt.plot(fpr, tpr, label=model[1] + ' - AUC= ' + str(auc))
  count = count + 1

plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC (trained on src, evaluated on src)')
plt.legend(loc="lower right")
plt.show()

plot_cfm(y,y_pred_uu,'Unparameterized-Unweighted')
plot_cfm(y,y_pred_uw,'Unparameterized-Reweighted')
plot_cfm(y,y_pred_pu,'Parameterized-Unweighted')
plot_cfm(y,y_pred_pw,'Parameterized-Reweighted')
