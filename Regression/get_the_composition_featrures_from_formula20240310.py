#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re, numpy as np, os, sys, pandas
import data_utils
import pandas as pd
from pymatgen import Composition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf  # 增加的是composition属性
from matminer.utils.conversions import str_to_composition
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# In[2]:


def get_the_composition_featrures_from_formula(df):

# In[3]:

    
    #改名
    df = df.rename(columns={'Formula': 'formula'})
    
    
    # In[4]:
    
    
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              #cf.ElementProperty.from_preset("matminer"),
                                              #cf.ElementProperty.from_preset("matscholar_el"),
                                              cf.ValenceOrbital(props=['avg'])]) #, cf.IonProperty(fast=True),
                                             
                                              #cf.CohesiveEnergy(mapi_key='zuTyaqQSZ18zjN1nzGVu')]
                                            
    '''
    feature_calculators = MultipleFeaturizer([
                                              cf.ValenceOrbital(props=['avg'])]
                                              
    '''
    
    
    # In[5]:
    
    
    feature_labels = feature_calculators.feature_labels()
    
    
    # In[6]:
    
    
    print (feature_labels, len(feature_labels))
    
    
    
    #%% np.nan 替换成 NaN 系统错误把 nan 识别成 nan float object
    MAPPING = {np.nan: 'NaN'}
    df['formula']= df['formula'].replace(MAPPING)#'pretty_formula'
    #% 防止后面get_fraction 因为误认nan为float报错。
    df['formula'].astype(str)
    
    
    # In[8]:
    
    
    df['comp_obj'] = df['formula'].apply(lambda x: Composition(x))  
    # 要现场加这个，不然下面featurize_dataframe报错 attribute
    #error，说是‘str’
    
    
    # In[9]:
    
    df=feature_calculators.featurize_dataframe(df, col_id='comp_obj',ignore_errors=True) #
    
    # In[10]:
    
    
    print(df.columns)
    
    
    # In[11]:
    
    
    df.head()
    
    
    
    
    # In[12]:
    
    
    df[feature_labels].shape
    
    
    # In[13]:
    
    
    # 删除没有特征的值
    '''
    row_before=len(df)
    df = df[~df[feature_labels].isnull().any(axis=1)]
    print (df.shape)
    df.reset_index(drop=True, inplace=True) # 删完东西重新排序，免得出错
    print('Removed duplicates %d/%d entries'%(row_before - len(df), row_before))
    '''
    
    return df

# In[14]:
if __name__ == "__main__": 
    
    #df=pd.read_csv('../dataset/MaterialProject_nelements_1~9_20200521_stability.csv') # 增加数据集，八万多stability数据
     # 增加数据集，增加原子序数为input，八万多stability数据
    #df=pd.read_csv('../dataset/MaterialProject_nelements_1~9_20200525_elasticity _atomicnum_in.csv')
    #df=pd.read_csv(r'../dataset/Hf_Ta_W_elements_combination.csv')
    #df=pd.read_csv('MPDS_MACHINE_LEARNING__all_elements_temperature for congruent melting_0601_cleaning.csv')
    #df=pd.read_csv('quinary_B_C_N_Ti_Zr_Nb_Hf_Ta_W_Re_Os_Mo_max_natoms10_combination.csv')
    df=pd.read_csv('../dataset/C_Ta_Hf_elements_combination.csv')
    #df=pd.read_csv('df_melting_all_2021-02-02_10_11_42count3678_test.csv')#UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb9 in position 9: invalid start byte #有一些格式
    #df=pd.read_csv('MPDS_peer-reviewed_thermal_20210303_cleaning_new.csv')
    
    df_get=get_the_composition_featrures_from_formula(df)
    df_get.to_csv('..dataset/C_Ta_Hf_elements_combination_3features123.csv',index=False)


# In[ ]:




