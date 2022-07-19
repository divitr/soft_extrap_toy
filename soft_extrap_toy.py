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
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import activations
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
import math


def generate_data(size):
  # generating pt (target)
  # . normal distribution with mean 500 and std 50
  pt_tar = norm.rvs(500, 50, size=size)
  label_tar = list()
  for pt in pt_tar:
    label_tar.append('target')
  pt_tar_labelled = zip(pt_tar, label_tar)

  # generating pt (source)
  # . truncated normal distribution with mean 0, std 100, truncated between 0 and 1000
  clip_a, clip_b, mean, std = 0, 1500, 0, 100
  a, b = (clip_a - mean) / std, (clip_b - mean) / std
  pt_src = truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
  label_src = list()
  for pt in pt_src:
    label_src.append('source')
  pt_src_labelled = zip(pt_src, label_src)

  # generating pt (background)
  # . exponential distribution with loc 0 and scale 200
  pt_back = expon.rvs(0, 200, size=size)
  label_back = list()
  for pt in pt_back:
    label_back.append('background')
  pt_back_labelled = zip(pt_back, label_back)

  pt_list = list(pt_tar_labelled) + list(pt_src_labelled) + \
                 list(pt_back_labelled)

  # construct dataframe with pt values
  df = pd.DataFrame(pt_list, columns=['pt', 'label'])

  # one-hot encode labels
  df = pd.get_dummies(df)

  # generate features
  f1_list = list()
  f2_list = list()
  for i in range(len(df)):
    pt = df.iloc[i, df.columns.get_loc('pt')]
    if (df.iloc[i, df.columns.get_loc('label_background')] == 1):
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


def plot_pt(train_df):
    df_src = train_df.loc[(train_df['label_source'] == 1)]
    df_back = train_df.loc[(train_df['label_background'] == 1)]
    df_tar = train_df.loc[(train_df['label_target'] == 1)]
    bins = np.linspace(0, 1500, 100)
    n_src, bins_src, patches_src = plt.hist(
        df_src['pt'].to_numpy(), bins, density=True, histtype='step', label='Source')
    n_tar, bins_tar, patches_tar = plt.hist(
        df_tar['pt'].to_numpy(), bins, density=True, histtype='step', label='Target')
    n_back, bins_back, patches_back = plt.hist(df_back['pt'].to_numpy(
    ), bins, density=True, histtype='step', label='Background')
    plt.legend(loc='best')
    plt.show()

    # network reweighting
    df = train_df.loc[(train_df['label_source'] == 1) |
                       (train_df['label_background'] == 1)]

    X = df['pt'].to_numpy()
    y = df['label_background'].to_numpy()
    X_src = df.loc[(df['label_source'] == 1)]['pt'].to_numpy()

    reweighting_model = keras.Sequential()
    reweighting_model.add(layers.Dense(20, input_dim=1, activation='relu'))
    reweighting_model.add(layers.Dense(40, activation='relu'))
    reweighting_model.add(layers.Dense(1, activation='sigmoid'))
    reweighting_model.compile(optimizer=Adam(
        learning_rate=.001), loss='mean_squared_error', metrics=['accuracy'])
    reweighting_model.fit(X, y, epochs=20, batch_size=50,
                          verbose=1, shuffle=True)

    y_pred = reweighting_model.predict(X_src)
    weights = np.divide(y_pred, (1-y_pred))

    weights_nn = []
    for w in weights:
      weights_nn.append(w[0])
    train_df = train_df.loc[(train_df['label_source'] == 1) | (
        train_df['label_background'] == 1)]
    src_df = train_df.loc[(train_df['label_source'] == 1)]
    back_df = train_df.loc[(train_df['label_background'] == 1)]
    src_df['weight_nn'] = weights_nn
    back_df['weight_nn'] = 1.0
    train_df = pd.concat([src_df, back_df], ignore_index=True, sort=False)
    class_weights_nn = {
        0: size/sum(weights_nn),
        1: 1
    }

    # manual reweighting
    # . compute bin weights
    weights = list()
    for i in range(0, 99):
      weights.append(n_back[i]/n_src[i])

    # . assign weights for each pt value
    train_df['weight'] = 1.0  # default weight
    for i in range(train_df.shape[0]):
      if train_df.iloc[i, train_df.columns.get_loc('label_source')] == 1:
        for j in range(0, len(bins_src)):
          if (bins_src[j] < train_df.iloc[i, 0] <= bins_src[j+1]):  # is src and is in bin size
            train_df.iloc[i, train_df.columns.get_loc('weight')] = weights[j]
            break

    df_src = train_df.loc[(train_df['label_source'] == 1)]
    df_back = train_df.loc[(train_df['label_background'] == 1)]
    df_tar = train_df.loc[(train_df['label_target'] == 1)]

    sum_src = 0
    for i in range(len(df_src)):
      sum_src = sum_src + df_src.iloc[i, df_src.columns.get_loc('weight')]

    class_weights = {
        0: size/sum_src,
        1: 1
    }
    return train_df, class_weights_nn, class_weights

def train_models(train_df,class_weights_nn,class_weights):     #output 1 means background
        train_df = shuffle(train_df)
        
        # UR - NN
        X = train_df[['f1','f2']].to_numpy().reshape(-1,2)
        y = train_df[['label_background']].to_numpy()

        u_r_nn = keras.Sequential()
        u_r_nn.add(layers.Dense(16,activation='relu',input_shape = (2,)))
        u_r_nn.add(layers.Dense(8,activation='relu'))
        u_r_nn.add(layers.Dense(1,activation='sigmoid'))
        u_r_nn.compile(optimizer=Adam(learning_rate=.001),loss='mean_squared_error',metrics=['accuracy'])
        u_r_nn.fit(x=X,y=y,validation_split=.1,batch_size=16,epochs=10,shuffle=True,verbose=1,sample_weight=train_df['weight_nn'].to_numpy(),class_weight=class_weights_nn)
        
        # UR - Manual
        # output 1 means background
        X = train_df[['f1','f2']].to_numpy().reshape(-1,2)
        y = train_df[['label_background']].to_numpy()

        u_r_man = keras.Sequential()
        u_r_man.add(layers.Dense(16,activation='relu',input_shape = (2,)))
        u_r_man.add(layers.Dense(8,activation='relu'))
        u_r_man.add(layers.Dense(1,activation='sigmoid'))
        u_r_man.compile(optimizer=Adam(learning_rate=.001),loss='mean_squared_error',metrics=['accuracy'])
        u_r_man.fit(x=X,y=y,validation_split=.1,batch_size=16,epochs=10,shuffle=True,verbose=1,sample_weight=train_df['weight'].to_numpy(),class_weight=class_weights)
        
        # PR - NN
        X = train_df[['f1','f2','pt']].to_numpy().reshape(-1,3)
        y - train_df[['label_background']].to_numpy()

        p_r_nn = keras.Sequential()
        p_r_nn.add(layers.Dense(16,activation='relu',input_shape = (3,)))
        p_r_nn.add(layers.Dense(8,activation='relu'))
        p_r_nn.add(layers.Dense(1,activation='sigmoid'))
        p_r_nn.compile(optimizer=Adam(learning_rate=.001),loss='mean_squared_error',metrics=['accuracy'])
        p_r_nn.fit(x=X,y=y,validation_split=.1,batch_size=16,epochs=10,shuffle=True,verbose=1,sample_weight=train_df['weight_nn'].to_numpy(),class_weight=class_weights_nn)
        
        # PR - Manual
        X = train_df[['f1','f2','pt']].to_numpy().reshape(-1,3)
        y - train_df[['label_background']].to_numpy()

        p_r_man = keras.Sequential()
        p_r_man.add(layers.Dense(16,activation='relu',input_shape = (3,)))
        p_r_man.add(layers.Dense(8,activation='relu'))
        p_r_man.add(layers.Dense(1,activation='sigmoid'))
        p_r_man.compile(optimizer=Adam(learning_rate=.001),loss='mean_squared_error',metrics=['accuracy'])
        p_r_man.fit(x=X,y=y,validation_split=.1,batch_size=16,epochs=10,shuffle=True,verbose=1,sample_weight=train_df['weight'].to_numpy(),class_weight=class_weights)
        
        return u_r_nn, u_r_man, p_r_nn, p_r_man
    
def evaluate(test_df, u_nn, u_man, p_nn, p_man):
    X = test_df[['f1','f2']].to_numpy().reshape(-1,2)
    y = test_df['label_background'].to_numpy()
    
    y_pred = u_nn.predict(X)
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    u_nn_auc = roc_auc_score(y, y_pred)
    plt.plot(fpr, tpr, label='Unparameterized (NN Reweighted) - AUC= ' + str(u_nn_auc))
    
    y_pred = u_man.predict(X)
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    u_man_auc = roc_auc_score(y, y_pred)
    plt.plot(fpr, tpr, label='Unparameterized (Manually Reweighted) - AUC= ' + str(u_man_auc))
    
    X = test_df[['f1','f2','pt']].to_numpy().reshape(-1,3)
    y = test_df['label_background'].to_numpy()
    
    y_pred = p_nn.predict(X)
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    p_nn_auc = roc_auc_score(y, y_pred)
    plt.plot(fpr, tpr, label='Parameterized (NN Reweighted) - AUC= ' + str(p_nn_auc))
    
    y_pred = p_man.predict(X)
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    p_man_auc = roc_auc_score(y, y_pred)
    plt.plot(fpr, tpr, label='Parameterized (Manually Reweighted) - AUC= ' + str(p_man_auc))
    
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC (trained on src, evaluated on tar) - ' + str(size) + ' src samples')
    plt.legend(loc="lower right")
    plt.show()
  
    return u_nn_auc, u_man_auc, p_nn_auc, p_man_auc
  
sizes = [100, 500, 1000, 2000, 5000, 10000]
u_nn_avg = []
u_man_avg = []
p_nn_avg = []
p_man_avg = []

u_nn_err = []
u_man_err = []
p_nn_err = []
p_man_err = []

u_nn_auc_ls_g = []
u_man_auc_ls_g = []
p_nn_auc_ls_g = []
p_man_auc_ls_g = []

test_df = generate_data(10000)

for size in sizes:
    print(str(size)+'-----------------------------------------------------------------------------')
    train_df = generate_data(size)
    train_df, class_weights_nn, class_weights = plot_pt(train_df)
    models = []
    for i in range(5):
      u_r_nn, u_r_man, p_r_nn, p_r_man = train_models(train_df, class_weights_nn, class_weights)
      models.append((u_r_nn,u_r_man,p_r_nn,p_r_man))
    u_nn_auc_ls = []
    u_man_auc_ls = []
    p_nn_auc_ls = []
    p_man_auc_ls = []
    for nns in models:
      u_r_nn = nns[0]
      u_r_man = nns[1]
      p_r_nn = nns[2]
      p_r_man = nns[3]
      u_nn_auc, u_man_auc, p_nn_auc, p_man_auc = evaluate(test_df, u_r_nn, u_r_man, p_r_nn, p_r_man)
      u_nn_auc_ls.append(u_nn_auc)
      u_man_auc_ls.append(u_man_auc)
      p_nn_auc_ls.append(p_nn_auc)
      p_man_auc_ls.append(p_man_auc)
      u_nn_auc_ls_g.append(u_nn_auc)
      u_man_auc_ls_g.append(u_man_auc)
      p_nn_auc_ls_g.append(p_nn_auc)
      p_man_auc_ls_g.append(p_man_auc)
    u_nn_avg.append(sum(u_nn_auc_ls)/len(u_nn_auc_ls))
    u_man_avg.append(sum(u_man_auc_ls)/len(u_man_auc_ls))
    p_nn_avg.append(sum(p_nn_auc_ls)/len(p_nn_auc_ls))
    p_man_avg.append(sum(p_man_auc_ls)/len(p_man_auc_ls))

    u_nn_err.append(np.std(u_nn_auc_ls)/math.sqrt(5))
    u_man_err.append(np.std(u_man_auc_ls)/math.sqrt(5))
    p_nn_err.append(np.std(p_nn_auc_ls)/math.sqrt(5))
    p_man_err.append(np.std(p_man_auc_ls)/math.sqrt(5))

plt.errorbar(sizes,u_nn_avg,yerr=u_nn_err,label='Unparameterized NN Reweighted')
plt.errorbar(sizes,u_man_avg,yerr=u_man_err,label='Unparameterized Manually Reweighted')
plt.errorbar(sizes,p_nn_avg,yerr=p_nn_err,label='Parameterized NN Reweighted')
plt.errorbar(sizes,p_man_avg,yerr=p_man_err,label='Parameterized Manually Reweighted')
plt.title('Network Performance vs Training Set Size')
plt.xlabel('Number of src Samples in Training Set')
plt.ylabel('AUC')
plt.legend(loc='best')
plt.show()

x, y = map(list, zip(*u_nn_auc_ls_g))
plt.scatter(x,y,marker="o")
plt.xlabel('Number of src Samples in Training Set')
plt.ylabel('AUC')
plt.title('Unparameterized NN Reweighted')
plt.show()

x, y = map(list, zip(*u_man_auc_ls_g))
plt.scatter(x,y,marker="o")
plt.xlabel('Number of src Samples in Training Set')
plt.ylabel('AUC')
plt.title('Unparameterized Manually Reweighted')
plt.show()

x, y = map(list, zip(*p_nn_auc_ls_g))
plt.scatter(x,y,marker="o")
plt.xlabel('Number of src Samples in Training Set')
plt.ylabel('AUC')
plt.title('Parameterized NN Reweighted')
plt.show()

x, y = map(list, zip(*p_man_auc_ls_g))
plt.scatter(x,y,marker="o")
plt.xlabel('Number of src Samples in Training Set')
plt.ylabel('AUC')
plt.title('Parameterized Manually Reweighted')
plt.show()

sizes = [100, 500, 1000, 2000, 5000, 10000]

y = u_nn_auc_ls_g
x = [100,100,100,100,100,500,500,500,500,500,1000,1000,1000,1000,1000,2000,2000,2000,2000,2000,5000,5000,5000,5000,5000,10000,10000,10000,10000,10000]
plt.figure(1)
plt.subplot(2,2,1)
plt.scatter(x,y,marker="o")
plt.plot(sizes,u_nn_avg,label='Average')
plt.xlabel('Number of src Samples in Training Set')
plt.ylabel('AUC')
plt.title('Unparameterized NN Reweighted')
plt.legend(loc='lower right')

plt.subplot(2,2,2)
y = u_man_auc_ls_g
plt.scatter(x,y,marker="o")
plt.plot(sizes,u_man_avg,label='Average')
plt.xlabel('Number of src Samples in Training Set')
plt.ylabel('AUC')
plt.title('Unparameterized Manually Reweighted')
plt.legend(loc='lower right')

plt.subplot(2,2,3)
y = p_nn_auc_ls_g
plt.scatter(x,y,marker="o")
plt.plot(sizes,p_nn_avg,label='Average')
plt.xlabel('Number of src Samples in Training Set')
plt.ylabel('AUC')
plt.title('Parameterized NN Reweighted')
plt.legend(loc='lower right')

plt.subplot(2,2,4)
y = p_man_auc_ls_g
plt.scatter(x,y,marker="o")
plt.plot(sizes,p_man_avg,label='Average')
plt.xlabel('Number of src Samples in Training Set')
plt.ylabel('AUC')
plt.title('Parameterized Manually Reweighted')
plt.legend(loc='lower right')
plt.show()
