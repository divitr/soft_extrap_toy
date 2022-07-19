from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from cv2 import threshold
from tensorflow import keras
from sklearn.metrics import plot_roc_curve,roc_curve,auc,roc_auc_score
from sklearn.metrics import accuracy_score
#from sklearm.utils import shuffle
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from math import isnan
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow_decision_forests as tfdf
import csv
from scipy.stats import truncnorm
from scipy.stats import norm
def generate_data(num:int, class_name: str, para_bool: int,path_store = None, reweight_method = 'NN'):
    np.random.seed(int(time.time() + 10*np.random.rand(1)+3))
    def generate_pt(num: int):
        np.random.seed(int(time.time() + 10*np.random.rand(1)+3))
        def generate_pt_(num: int):
            pt_target_loc = []
            pt_source_loc = []
            pt_background = []
            if class_name == 'source':                
                # for i in range(int(num)):
                #     temp =  np.random.normal(loc=0,scale = 100,size = 1)
                #     if temp < 0:
                #         temp = -temp
                #     pt_source_loc = np.append(pt_source_loc,temp)
                clip_a, clip_b, mean, std = 0, 1500, 0, 100
                a, b = (clip_a - mean) / std, (clip_b - mean) / std
                pt_source_loc = truncnorm.rvs(a, b, loc = mean, scale = std, size = num)
            if class_name == 'target':                
                for i in range(int(num)):
                    temp =  np.random.normal(loc=500,scale = 50,size = 1)
                    if temp < 0:
                        temp = -temp
                    pt_target_loc = np.append(pt_target_loc,temp)
                # pt_target_loc = norm.rvs(500, 50, size = num)
            
            for i in range(int(num)):
                temp =  np.random.exponential(scale = 200, size = 1)
                #temp =  np.random.normal(loc=0, scale = 8,size = 1)
                if temp < 0:
                    temp = -temp
                pt_background = np.append(pt_background,temp)
            
            return pt_source_loc, pt_target_loc, pt_background
        def generate_dataframe_pt(pt_source_loc:np.array, pt_background:np.array):
            id = np.ones([len(pt_source_loc),1])
            id = id.astype(int)
            id = id.flatten()
            id = pd.DataFrame({'class_id':id})
            id_b = np.zeros([len(pt_background),1])
            id_b = id_b.astype(int)
            id_b = id_b.flatten()
            id_b = pd.DataFrame({'class_id':pd.Series(id_b)})
            class_id = pd.concat([id,id_b])
            class_id.reset_index(drop=True, inplace=True)

        
            pt_source_loc = pd.DataFrame({'pt': pd.Series(pt_source_loc)})
            pt_background = pd.DataFrame({'pt': pd.Series(pt_background)})
            
            pt = pd.concat([pt_source_loc, pt_background])
            pt.reset_index(drop=True, inplace=True)
            #f2 = pd.concat([f2_source, f2_background])
            f = pd.DataFrame({'pt':pt['pt'],'class_id': class_id['class_id']}, index=range(len(class_id)))
        
            return f       
            
        
        pt_source, pt_target, pt_background = generate_pt_(num)
        
        
        return pt_source, pt_target, pt_background
    def generate_f1(pt_source: np.array, pt_target: np.array , pt_background: np.array, num): 
        theta = np.random.normal(loc = 0, scale = math.pi/3, size = len(pt_background))
        f1_background = np.power(pt_background,0.2)
        f1_background = f1_background - np.multiply(pt_background,(np.sin(theta)-0.5))
        f1_background = f1_background +50
        f1_source = []
        f1_target = []
        if class_name == 'source':
            theta = np.random.normal(loc = 0, scale = math.pi/3, size = len(pt_source))
            f1_source = np.power(pt_source,0.2)
            f1_source = np.add(f1_source, (1)*np.multiply(pt_source,(np.sin(theta)-0.5)))
            f1_source = f1_source +50
        if class_name == 'target':
            theta = np.random.normal(loc = 0, scale = math.pi/3, size = len(pt_target))
            f1_target = np.power(pt_target,0.2)
            f1_target = np.add(f1_target, (1)*np.multiply(pt_target,(np.sin(theta)-0.5)))
            f1_target = f1_target + 50
        return  f1_source, f1_target, f1_background
    def generate_f2(pt_source: np.array, pt_target: np.array , pt_background: np.array, num): 
        np.random.seed(int(time.time()))
        theta = np.random.normal(loc = 0, scale = math.pi/3, size = len(pt_background))
        f2_background = np.power(pt_background,1.1)
        f2_background = np.add(f2_background, (-1)*np.multiply(pt_background,(np.cos(theta)-0.5)))
        f2_background = f2_background +5
        f2_source= []
        f2_target = []
        if class_name == 'source':
            theta = np.random.normal(loc = 0, scale = math.pi/3, size = len(pt_source))
            f2_source = np.power(pt_source,1.1)
            f2_source = np.add(f2_source, (1)*np.multiply(pt_source,(np.cos(theta)-0.5)))
            f2_source = f2_source +5
        if class_name == 'target':
            theta = np.random.normal(loc = 0, scale = math.pi/3, size = len(pt_target))
            f2_target = np.power(pt_target,1.1)
            f2_target = np.add(f2_target, (1)*np.multiply(pt_target,(np.cos(theta)-0.5)))
            f2_target = f2_target + 5
        return  f2_source, f2_target,f2_background
    def generate_dataframe(f1_source_:np.array, f1_background_:np.array, f2_source_:np.array, f2_background_:np.array, pt_source_:np.array, pt_background_):
        id = np.ones([len(pt_source_),1])
        id = id.astype(int)
        id = id.flatten()
        id = pd.DataFrame({'class_id':id})
        id_b = np.zeros([len(pt_background_),1])
        id_b = id_b.astype(int)
        id_b = id_b.flatten()
        id_b = pd.DataFrame({'class_id':pd.Series(id_b)})
        class_id = pd.concat([id,id_b])
        class_id.reset_index(drop=True, inplace=True)

        f1_source_ = pd.DataFrame({'x1': pd.Series(f1_source_)})
        f2_source_ = pd.DataFrame({'x2': pd.Series(f2_source_)})
        pt_source_ = pd.DataFrame({'pt': pd.Series(pt_source_)})
        #f1_source.insert(loc = 2, column = 'class_ID',value = 1)
        f1_background_ = pd.DataFrame({'x1': pd.Series(f1_background_)})
        f2_background_ = pd.DataFrame({'x2': pd.Series(f2_background_)})
        pt_background_ = pd.DataFrame({'pt': pd.Series(pt_background_)})
        #f1_background.insert(loc = 2, column = 'class_ID',value = 0)
        f1 = pd.concat([f1_source_, f1_background_])
        f1.reset_index(drop = True, inplace=True)
        f2 = pd.concat([f2_source_, f2_background_])
        f2.reset_index(drop = True, inplace=True)
        pt = pd.concat([pt_source_, pt_background_])
        pt.reset_index(drop=True, inplace=True)
        #f2 = pd.concat([f2_source, f2_background])
        f = pd.DataFrame({'x1':f1['x1'],'x2':f2['x2'],'pt':pt['pt'],'class_id': class_id['class_id']}, index=range(len(class_id)))
        return f
    pt_source, pt_target, pt_background = generate_pt(num)
    f1_source, f1_target,f1_background = generate_f1(pt_source, pt_target, pt_background, num)
    f2_source, f2_target,f2_background = generate_f2(pt_source, pt_target, pt_background, num)
    
    
    weight = []#if 'target', weight = []
    if class_name == 'source':
        f_train = generate_dataframe(f1_source, f1_background, f2_source, f2_background, pt_source, pt_background)
        y_train = f_train.pop('class_id')

        pt_source, pt_target, pt_background = generate_pt(num)
        f1_source, f1_target,f1_background = generate_f1(pt_source, pt_target, pt_background, num)        
        f2_source, f2_target,f2_background = generate_f2(pt_source, pt_target, pt_background, num)
        f = generate_dataframe(f1_source, f1_background, f2_source, f2_background, pt_source, pt_background)
        y = f.pop('class_id')
        if reweight_method == 'NN':
            model_assist = Sequential()
            model_assist.add(Dense(20, input_dim=1, activation='relu'))
            model_assist.add(Dense(40, activation='relu'))
            model_assist.add(Dense(1, activation='sigmoid'))
            model_assist.compile(loss='mean_squared_error', optimizer=Adam(learning_rate = 0.001), metrics=['accuracy'])
            
            model_assist.fit(pd.DataFrame({'pt':f_train['pt']}), y_train, epochs=16, batch_size=20, verbose=1,shuffle=True)
            y_pred_keras = model_assist.predict(pd.DataFrame({'pt':f['pt']}))
            weight_sb = np.divide(y_pred_keras,1-y_pred_keras)
            #weight_sb = weight_sb / np.max(weight_sb)
            weight_sb[0:num] = weight_sb[0:num] / np.sum(weight_sb[0:num]) * np.sum(weight_sb[num:2 * num])
            weight = weight_sb
            '''
            model_assist.fit(pd.DataFrame({'pt':f_train['pt']}), y_train, epochs=100, batch_size=50, verbose=1,shuffle=True,sample_weight=weight)
            y_pred_keras = model_assist.predict(pd.DataFrame({'pt':f['pt']}))
            weight_sb = np.divide(y_pred_keras,1-y_pred_keras)
            #weight_sb = weight_sb / np.max(weight_sb)
            weight_sb[0:num] = weight_sb[0:num] / np.sum(weight_sb[0:num]) * np.sum(weight_sb[num:2 * num])
            weight = weight_sb
            '''
            n_src, bin_src = np.histogram(pt_source, bins=100, range=(0,1500), density=True)
            n_bkg, bin_bkg = np.histogram(pt_background, bins=100, range=(0,1500), density=True)
            vice_weight_sb = []
            vice_weight = []
            for i in range(0, 99):
                vice_weight_sb = np.append(vice_weight_sb, (n_bkg[i]/n_src[i]))
            
            for i in range(num):
                for j in range(len(vice_weight_sb)):
                    if pt_source[i] > bin_src[j] and pt_source[i] <= bin_src[j+1]:
                        vice_weight = np.append(vice_weight, vice_weight_sb[j])
            temp = np.sum(weight) / num
            for i in range(num):
                vice_weight = np.append(vice_weight, temp)
            for i in range(len(weight)):
                temp = weight[i]*vice_weight[i]
                weight[i] = temp
            weight = weight/np.sum(weight)
        
        if reweight_method == 'man':
            n_src, bin_src = np.histogram(pt_source, bins=100, range=(0,1500), density=True)
            n_bkg, bin_bkg = np.histogram(pt_background, bins=100, range=(0,1500), density=True)
            weight_sb = []
            for i in range(0, 99):
                weight_sb = np.append(weight_sb, (n_bkg[i]/n_src[i]))
            
            for i in range(num):
                for j in range(len(weight_sb)):
                    if pt_source[i] > bin_src[j] and pt_source[i] <= bin_src[j+1]:
                        weight = np.append(weight, weight_sb[j])
            temp = np.sum(weight) / num
            for i in range(num):
                weight = np.append(weight, temp)
        plt.figure(dpi=500)
        plt.hist(pt_background,bins = 100,range=(0,1500),weights=weight[num:2*num],density=True, histtype='step',label='background')

        plt.hist(pt_source,bins = 100,range=(0,1500),weights=weight[0:num],density = True, histtype='step', label='source') 
        #plt.hist(pt_background,bins = 100,range=(0,1500), density=True, histtype='step',label='background')
        #plt.hist(pt_source,bins = 100,range=(0,1500) ,density = True, histtype='step', label='source') 
        plt.ylim([0,0.018])
        plt.xlabel('Pt')
        plt.ylabel('probability density')
        plt.title(str(num))
        plt.legend(loc='best')
        if path_store != None:
            plt.savefig(path_store)
        #plt.text(1200,0.010,str(num),color='black')
        plt.close()


        #plt.hist(f1_background,bins = 100,range=(0,1500),weights=weight[num:2*num],density=True, histtype='step')

        #plt.hist(f1_source,bins = 100,range=(0,1500),weights=weight[0:num],density = True, histtype='step')  
        
    #     plt.show()  

    # plt.scatter(f1_source,f2_source,s=10,c='r',marker='o')
    # plt.scatter(f1_target,f2_target,s=10,c='b',marker='v')
    # plt.scatter(f1_background,f2_background,s=10,c='g',marker='*')  
    # plt.show()

    
        
       
        
    if class_name == 'target':
        f = generate_dataframe(f1_target, f1_background, f2_target, f2_background,pt_target,pt_background)
        y = f.pop('class_id')
        #weight = np.ones(len(f1_source))
    
    if para_bool == 0:
        f.pop('pt')

    return f, y ,weight#if 'target', weight = []
def build_model_unpara():
    model = Sequential()
    model.add(Dense(12, input_dim=2, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=.001), metrics=['accuracy'])
    return model
def build_model_para():
    model = Sequential()
    model.add(Dense(12, input_dim=3, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=.001), metrics=['accuracy'])
    return model
if __name__ == "__main__":
    header = ['unpara_tt','unpara_st', 'para_tt', 'para_ts','unpara_tt_std','unpara_st_std', 'para_tt_std', 'para_ts_std']

    with open('/mnt/d/DW_project/num/0718/NN/outcome_auc.csv', 'a+', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
    with open('/mnt/d/DW_project/num/0718/man/outcome_auc.csv', 'a+', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
    num_list = [0.05, 0.1,0.5, 1,5, 10]#[0.005, 0.01, 0.05,0.1,0.5,1,5,10,50,100] 
    loop_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,15,16,17,18,19]
    #num_list = [ 8, 20]
    num_list = np.multiply(num_list , 1000)
    num_list.astype(int)

    f_source_test, y_f_source_test,weight_src_te = generate_data(100000, 'target',0)
    time.sleep(1)

    f_target_test,y_f_target_test, weight_tar_tr = generate_data(100000, 'target',0)
    time.sleep(1)

    f_source_test_P, y_f_source_test_P,weight_src_te_P = generate_data(100000, 'target',1)
    time.sleep(1)

    f_target_test_P,y_f_target_test_P, weight_tar_tr_P = generate_data(100000, 'target',1)
    time.sleep(1)
    for num in num_list:
        auc_tt_list = []
        auc_st_list = []
        auc_p_tt_list = []
        auc_p_st_list = []
        for i_loop in loop_list:#,5,6,7,8,9,10,11,12,13,14]:
            try:
                num = int(num)
                i_loop = int(i_loop)
                store_path_NN = '/mnt/d/DW_project/num/0718/'+str(i_loop)+'weight'+str(num)+'NN'+'.png'
                
                ##########################################################
                #un-para
                f_source_train, y_f_source_train, weight_src_tr = generate_data(num, 'source',0,store_path_NN)
                # plt.hist(f_source_train['pt'][0:num], bins=100, range=[0,500],density = True ,color ='r',histtype='step',label = 'pt_source', weights=  weight_src_tr[0:num])
                # plt.hist(f_source_train['pt'][num:2*num], bins=100, range=[0,500] ,density = True,color ='b',histtype='step',label = 'pt_target', weights=  weight_src_tr[num:2*num])
                time.sleep(1)
                
                f_target_train,y_f_target_train, weight_tar_tr = generate_data(num, 'target',0)
                time.sleep(1)

                unP_model_st = build_model_unpara()
                unP_model_st.fit(f_source_train, y_f_source_train, epochs=10, batch_size=20, verbose=1,shuffle=True,sample_weight = weight_src_tr,validation_split=0.1)
                y_unP_st = unP_model_st.predict(f_source_test)
                fpr_st, tpr_st, thresholds_tt = roc_curve(y_f_source_test, y_unP_st)
                auc_st = auc(fpr_st, tpr_st)
                
                unP_model_tt = build_model_unpara()
                unP_model_tt.fit(f_target_train, y_f_target_train, epochs=10, batch_size=20, verbose=1,shuffle=True,validation_split=0.1)
                y_unP_tt = unP_model_tt.predict(f_target_test)
                fpr_tt, tpr_tt, thresholds_tt = roc_curve(y_f_target_test, y_unP_tt)
                auc_tt = auc(fpr_tt, tpr_tt)

                # if auc_tt < 0.5 :
                #     auc_tt = 1 - auc_tt
                #     temp = tpr_tt
                #     tpr_tt = fpr_tt 
                #     fpr_tt = temp
                plt.figure()
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr_st, tpr_st, label='source-target (area = {:.3f})'.format(auc_st))
                auc_st_list = np.append(auc_st_list,auc_st)
                plt.plot(fpr_tt, tpr_tt, label='target-target (area = {:.3f})'.format(auc_tt),linestyle = 'dashed')
                auc_tt_list = np.append(auc_tt_list,auc_tt)
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('(unpara)ROC curve')
                plt.legend(loc='best')
                plt.savefig('/mnt/d/DW_project/num/0718/NN/' + str(num) + 'unpara' + str(i_loop) + '.png')
                plt.close()




                ##########################################################
                #para
                f_source_train, y_f_source_train, weight_src_tr = generate_data(num, 'source',1)
                # plt.hist(f_source_train['pt'][0:num], bins=100, range=[0,500],density = True ,color ='r',histtype='step',label = 'pt_source', weights=  weight_src_tr[0:num])
                # plt.hist(f_source_train['pt'][num:2*num], bins=100, range=[0,500] ,density = True,color ='b',histtype='step',label = 'pt_target', weights=  weight_src_tr[num:2*num])
                time.sleep(1)
                
                f_target_train,y_f_target_train, weight_tar_tr = generate_data(num, 'target',1)
                time.sleep(1)

                P_model_st = build_model_para()
                P_model_st.fit(f_source_train, y_f_source_train, epochs=10, batch_size=20, verbose=1,shuffle=True,sample_weight = weight_src_tr,validation_split=0.1)
                y_P_st = P_model_st.predict(f_source_test_P)
                fpr_st, tpr_st, thresholds_tt = roc_curve(y_f_source_test_P, y_P_st)
                auc_st = auc(fpr_st, tpr_st)

                P_model_tt = build_model_para()
                P_model_tt.fit(f_target_train, y_f_target_train, epochs=10, batch_size=20, verbose=1,shuffle=True,validation_split=0.1)
                y_P_tt = P_model_tt.predict(f_target_test_P)
                fpr_tt, tpr_tt, thresholds_tt = roc_curve(y_f_target_test_P, y_P_tt)
                auc_tt = auc(fpr_tt, tpr_tt)

                # if auc_tt < 0.5 :
                #     auc_tt = 1 - auc_tt
                #     temp = tpr_tt
                #     tpr_tt = fpr_tt 
                #     fpr_tt = temp
                plt.figure(dpi=500)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr_st, tpr_st, label='source-target (area = {:.3f})'.format(auc_st))
                auc_p_st_list = np.append(auc_p_st_list,auc_st)
                plt.plot(fpr_tt, tpr_tt, label='target-target (area = {:.3f})'.format(auc_tt),linestyle = 'dashed')
                auc_p_tt_list = np.append(auc_p_tt_list,auc_tt)
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('(para)ROC curve')
                plt.legend(loc='best')
                plt.savefig('/mnt/d/DW_project/num/0718/NN/' + str(num) + 'para' + str(i_loop) + '.png')
                plt.close()
            except:
                continue;
        auc_row = [np.max(auc_tt_list),np.max(auc_st_list),np.max(auc_p_tt_list),np.max(auc_p_st_list),np.std(auc_tt_list),np.std(auc_st_list),np.std(auc_p_tt_list),np.std(auc_p_st_list)]
        with open('/mnt/d/DW_project/num/0718/NN/outcome_auc.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(auc_row)
    for num in num_list:
        auc_tt_list = []
        auc_st_list = []
        auc_p_tt_list = []
        auc_p_st_list = []
        for i_loop in loop_list:#,5,6,7,8,9,10,11,12,13,14]:
            try:
                store_path_man = '/mnt/d/DW_project/num/0718/'+str(i_loop)+'weight'+str(num)+'man'+'.png'
                num = int(num)
                i_loop = int(i_loop)
                ##########################################################
                #un-para
                f_source_train, y_f_source_train, weight_src_tr = generate_data(num, 'source',0,store_path_man,'man')
                # plt.hist(f_source_train['pt'][0:num], bins=100, range=[0,500],density = True ,color ='r',histtype='step',label = 'pt_source', weights=  weight_src_tr[0:num])
                # plt.hist(f_source_train['pt'][num:2*num], bins=100, range=[0,500] ,density = True,color ='b',histtype='step',label = 'pt_target', weights=  weight_src_tr[num:2*num])
                time.sleep(1)
                
                f_target_train,y_f_target_train, weight_tar_tr = generate_data(num, 'target',0)
                time.sleep(1)

                unP_model_st = build_model_unpara()
                unP_model_st.fit(f_source_train, y_f_source_train, epochs=10, batch_size=20, verbose=1,shuffle=True,sample_weight = weight_src_tr,validation_split=0.1)
                y_unP_st = unP_model_st.predict(f_source_test)
                fpr_st, tpr_st, thresholds_tt = roc_curve(y_f_source_test, y_unP_st)
                auc_st = auc(fpr_st, tpr_st)
                
                unP_model_tt = build_model_unpara()
                unP_model_tt.fit(f_target_train, y_f_target_train, epochs=10, batch_size=20, verbose=1,shuffle=True,validation_split=0.1)
                y_unP_tt = unP_model_tt.predict(f_target_test)
                fpr_tt, tpr_tt, thresholds_tt = roc_curve(y_f_target_test, y_unP_tt)
                auc_tt = auc(fpr_tt, tpr_tt)

                # if auc_tt < 0.5 :
                #     auc_tt = 1 - auc_tt
                #     temp = tpr_tt
                #     tpr_tt = fpr_tt 
                #     fpr_tt = temp
                plt.figure(dpi=500)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr_st, tpr_st, label='source-target (area = {:.3f})'.format(auc_st))
                auc_st_list = np.append(auc_st_list,auc_st)
                plt.plot(fpr_tt, tpr_tt, label='target-target (area = {:.3f})'.format(auc_tt),linestyle = 'dashed')
                auc_tt_list = np.append(auc_tt_list,auc_tt)
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('(unpara)ROC curve')
                plt.legend(loc='best')
                plt.savefig('/mnt/d/DW_project/num/0718/man/' + str(num) + 'unpara' + str(i_loop) + '.png')
                plt.close()




                ##########################################################
                #para
                f_source_train, y_f_source_train, weight_src_tr = generate_data(num, 'source',1,'man')
                # plt.hist(f_source_train['pt'][0:num], bins=100, range=[0,500],density = True ,color ='r',histtype='step',label = 'pt_source', weights=  weight_src_tr[0:num])
                # plt.hist(f_source_train['pt'][num:2*num], bins=100, range=[0,500] ,density = True,color ='b',histtype='step',label = 'pt_target', weights=  weight_src_tr[num:2*num])
                time.sleep(1)
                
                f_target_train,y_f_target_train, weight_tar_tr = generate_data(num, 'target',1)
                time.sleep(1)

                P_model_st = build_model_para()
                P_model_st.fit(f_source_train, y_f_source_train, epochs=10, batch_size=20, verbose=1,shuffle=True,sample_weight = weight_src_tr,validation_split=0.1)
                y_P_st = P_model_st.predict(f_source_test_P)
                fpr_st, tpr_st, thresholds_tt = roc_curve(y_f_source_test_P, y_P_st)
                auc_st = auc(fpr_st, tpr_st)

                P_model_tt = build_model_para()
                P_model_tt.fit(f_target_train, y_f_target_train, epochs=10, batch_size=20, verbose=1,shuffle=True,validation_split=0.1)
                y_P_tt = P_model_tt.predict(f_target_test_P)
                fpr_tt, tpr_tt, thresholds_tt = roc_curve(y_f_target_test_P, y_P_tt)
                auc_tt = auc(fpr_tt, tpr_tt)

                # if auc_tt < 0.5 :
                #     auc_tt = 1 - auc_tt
                #     temp = tpr_tt
                #     tpr_tt = fpr_tt 
                #     fpr_tt = temp
                plt.figure()
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr_st, tpr_st, label='source-target (area = {:.3f})'.format(auc_st))
                auc_p_st_list = np.append(auc_p_st_list,auc_st)
                plt.plot(fpr_tt, tpr_tt, label='target-target (area = {:.3f})'.format(auc_tt),linestyle = 'dashed')
                auc_p_tt_list = np.append(auc_p_tt_list,auc_tt)
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('(para)ROC curve')
                plt.legend(loc='best')
                plt.savefig('/mnt/d/DW_project/num/0718/man/' + str(num) + 'para' + str(i_loop) + '.png')
                plt.close()
            except:
                continue;
        auc_row = [np.max(auc_tt_list),np.max(auc_st_list),np.max(auc_p_tt_list),np.max(auc_p_st_list),np.std(auc_tt_list),np.std(auc_st_list),np.std(auc_p_tt_list),np.std(auc_p_st_list)]
        with open('/mnt/d/DW_project/num/0718/man/outcome_auc.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(auc_row)
'''
df=pd.read_excel('/mnt/d/DW_project/num/0629/results.xlsx')
sizes = df['number']
unpara_st = df['unpara_st']
para_ts = df['para_ts']
unpara_st_std = df['unpara_st_std'] / (5**0.5)
para_ts_std = df['para_ts_std'] / (5**0.5)
plt.figure(dpi=500)
sizes = [50,100,500,1000,5000,10000,20000,50000,100000]
unpara_st = [0.5444,0.5746,0.577,0.5992,0.62,0.608,0.639,0.671,0.635]
unpara_st_std=[0.271790802,0.136241844,0.083603828,0.085307444,0.027224989,0.02668033,0.0332,0.020137031,0.010034316]
unpara_st_std = unpara_st_std / np.sqrt(5)
para_ts=[0.5308,0.4392,0.5108,0.7382,0.7204,0.724,0.7482,0.739,0.7568]
para_ts_std = [0.226543947,0.227376692,0.196478396,0.075819259,0.056930133,0.107474648,0.038280021,0.015811388,0.02184857]
para_ts_std = para_ts_std / np.sqrt(5)
plt.errorbar(sizes,unpara_st,yerr=unpara_st_std,label='Unparameterized Manually Reweighted',ecolor='r',color='r',elinewidth=2,capsize=4)
plt.errorbar(sizes,para_ts,yerr=para_ts_std,label='Parameterized Manually Reweighted',ecolor='g',color='g',elinewidth=2,capsize=4)
plt.xscale('log')
plt.title('Network Performance vs Training Set Size')
plt.xlabel('Number of Source Samples in Training Set')
plt.ylabel('AUC')
plt.legend(loc='best')
plt.show()
'''
    


