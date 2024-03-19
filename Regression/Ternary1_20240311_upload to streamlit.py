# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 06:33:19 2020

@author: MSI1
"""
import tensorflow as tf #Version: 1.15.2

import numpy as np    
import matplotlib.pyplot as plt #from matplotlib import pyplot as plt
#import matplotlib.gridspec as gridspec
#from matplotlib.colors import LogNorm
import os
import math
import random
import time
from itertools import product
import pandas as pd
#import pickle as pkl
from pandas import DataFrame
#from pymatgen import Composition
#from scipy.stats import randint as sp_randint
#from matminer.figrecipes.plot import PlotlyFig
#from matminer.data_retrieval import retrieve_MDF
#from matminer.featurizers.base import MultipleFeaturizer
#from matminer.featurizers import composition as cf
#from matminer.featurizers.conversions import StrToComposition
import sklearn.preprocessing as sk
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
#from sklearn.base import TransformerMixin, BaseEstimator
#from sklearn.linear_model import LinearRegression
#from sklearn.svm import SVR, LinearSVR
#from sklearn.decomposition import PCA, NMF
#from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import RepeatedKFold, cross_val_score, cross_val_predict, train_test_split, GridSearchCV, RandomizedSearchCV,ShuffleSplit, KFold
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf  # 增加的是composition属性
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import datetime


#%%   旧 的combination生成的组合，原子数固定为10



def ternary_elements_product(elements_range=['H', 'Li', 'Be','Ti'],max_nelements=10):
    """
    生成三元化合物的全元素组合作为预测的输入。（包含一元，二元，三元，formula原子个数10及以内） 
    优化程序,所有循环都用dataframe, 但不用df.append()装元素，改用list.extend()
    添加formula和natoms
    1.创建只有列名的dataframe，没有第一行全为0。
    2. 从elements_in 提取3个元素排列。
    3. 把comb0('Li', 'Be', 'Ti')>'Li'1 'Be'2 'Ti'3+natoms+formula
    3.1 生成AxByCz的卡迪尔积，实际上是生成xyz三种系数的组合, (1000,6) 
    添加了'natioms','formula'列，'nelements'列
    Parameters
    ----------
    elements_range : 本文研究范围内的全部元素list,全组合应为elements_range=elements_in
        DESCRIPTION. The default is ['H', 'Li', 'Be','Ti'].
    max_nelements : formula原子个数10及以内
        DESCRIPTION. The default is 10.

    Returns
    -------
    生成三元化合物的全元素组合dataframe作为预测的输入。

    """
    import numpy as np
    import pandas as pd
    from itertools import product
    from itertools import combinations,permutations
    import time
    #测试才开global，免得影响别的variable
    #global comb, product0,product1,product2,product3,natoms_series
    
    #s = list(np.round(np.arange(0.1,1,0.1),2))
    #print(len(s))
    elements_range = elements_range#['H', 'Li', 'Be','Ti']  # 为了简单测试，此处用4个元素抽三个出来。
    print('elements_range',elements_range)
    num=3 # number of elements
    max_nelements=max_nelements # 原子个数10及以内
    
    # 1. 创建只有列名的dataframe，没有第一行全为0
    #df=pd.DataFrame({x:[0]   for x in elements_in}) # 生成新的elements_in全部元素列的dataframe,第一行全为0
    df_all=pd.DataFrame(columns=['formula','natoms','nelements']+elements_in) # 将df改名为df_all防止名字混淆
    print('Generat a dataframe contain',df_all.columns)
    total_list=[]
    #2. 从elements_in 提取3个元素排列。#没有重复的，适合排列元素，注意要用list转化combinations object
    comb=list(combinations(elements_range, num)) # 从elements_in元素中抽取三个元素,形成排列组合 #73150,用时1s
    #print('comb',comb)
       
    # 3. 把comb0('Li', 'Be', 'Ti')>'Li'1 'Be'2 'Ti'3+natoms+formula
    tt1 = time.time()
    n=0
    for comb0 in comb:
         t1 = time.time()
         print('combination',n, comb0) # list ('Li', 'Be', 'Ti')
         n+=1
         #print(comb0[0],comb0[1],comb0[2]) #元素 'Li', 'Be', 'Ti'
        # 3.1 生成AxByCz的卡迪尔积，实际上是生成xyz三种系数的组合, (1000,6) 
        # 'A'不能是'Ti',不然会以'T'和'i'组合
         product0=pd.DataFrame(product('A',[x for x in np.round(np.arange(0,max_nelements,1),2)],   #[x for x in np.round(np.arange(0.1,1,0.1),2)] 生成0.1~0.9的list
                                    'B',[x for x in np.round(np.arange(0,max_nelements,1),2)],
                                    'C',[x for x in np.round(np.arange(0,max_nelements,1),2)]))# 
         product0.drop([0],inplace=True) #删除index=0的第一行，Li0 Be0 Ti0
         product0.columns=['A','x','B','y','C','z']
         
         product1=product0.query('x+y+z<=%d'%max_nelements)#,inplace=True) # 取原子个数10及以内 #(238，6)
         #把AxByCz的系数变为小数点（每个元素除以对应每行的和）
         product2=product1.drop(['A','B','C'],axis=1) #去除'A','B','C'列
         product2.columns=comb0 # 将x，y，z 改成'Li', 'Be', 'Ti'
         #product2[[x for x in comb0]] # 选取proudct2中的'Li', 'Be', 'Ti'列
         
         # 按行统计非0值数量，即为元素种类数量。nelements,要放在'natoms'前头
         product2['nelements']=(product2 != 0).astype(int).sum(axis=1)
         
         #只选取proudct2中的'Li', 'Be', 'Ti'列  #得到natoms
         product2['natoms'] = product2[[x for x in comb0]].apply(lambda x: x.sum(), axis=1)
         #把AxByCz的系数变为小数点（每个元素除以对应每行的和）
         product2[[x for x in comb0]]=product2[[x for x in comb0]].div(product2['natoms'], axis=0)
         # 只从元素列中除去重复系数.如Li 0 Be 0 Ti 1和 Li 0 Be 0 Ti 2  # 'natoms'列不管，所以用subset
         product2.drop_duplicates(subset=[x for x in comb0],keep='first', inplace=True)  #(205，3)
         # 得到formula H0Be0Li1
         product2['formula']=product2.apply(lambda x: 
                                            comb0[0]+str(int(x[comb0[0]]*x['natoms']))+
                                            comb0[1]+str(int(x[comb0[1]]*x['natoms']))+
                                            comb0[2]+str(int(x[comb0[2]]*x['natoms'])), axis=1)
         # 替换H0, Li0, Be0字符为空
         product2['formula']=product2.apply(lambda x:x['formula'].replace(comb0[0]+'0',''),axis=1)
         product2['formula']=product2.apply(lambda x:x['formula'].replace(comb0[1]+'0',''),axis=1)
         product2['formula']=product2.apply(lambda x:x['formula'].replace(comb0[2]+'0',''),axis=1)
        
         t_append = time.time()
         df_all=df_all.append(product2) # 将(205，3+2)>>>(205,77+2) # 将Li 0 Be 0 Ti 1 弄到77个元素的fraction中。
         #df_all=df_all.append(product2,ignore_index=True) # (205*77,77)     # 这个时间逐渐增加！！！主要影响程序！循环次数3000多的时候，已经是1s一个('Li', 'Be', 'Ti')组合
         comb0_list = df_all.values.tolist() #dataframe转换为列表
         total_list.extend(comb0_list) # 添加 list('Li', 'Be', 'Ti')的组合
         
         # 重新生成77个元素datafram列名供前面product2转换
         df_all=pd.DataFrame(columns=['formula','natoms','nelements']+elements_in) # 创建只有列名的dataframe，没有第一行全为0
         
         t2 = time.time()
         print('Dataframe Append time cost: %.6f'%(t2-t_append))
         # 用dataframe一个combination 7s>>0.01s
         print('Generate',comb0,'combination time cost: %.6f'%(t2-t1)) # 生成('Li', 'Be', 'Ti')组合为0.01 s

        
    #df_all=df_all.append(product2,ignore_index=True) # (205*77,77)
    df_all=pd.DataFrame(total_list)
    df_all.columns=['formula','natoms','nelements']+elements_in # 注意了[['formula','natoms','nelements']+elements_in]添个中括号，点开看是一样，实际上名字从formula变为（formula,）
    #print('df_all.columns[0]',df_all.columns[0]) 测试columns名字
    #df_all=df_all.append(df_all_list)
    df_all=df_all.fillna(0)  # 将NaN替换为0, append进去，有的数没有就会是nan.
    df_all.drop_duplicates('formula', keep='first', inplace=True)  #删除重复值。
    tt2 = time.time()
    print('Total time: %.6f'%(tt2-tt1)) # 6488.562333=1.8h
    
    
    import pandas as pd
    from pymatgen import Composition
    import data_utils
    df_get=df_all
    
    
    
    
    df_get['comp_obj'] = df_get['formula'].apply(lambda x: Composition(x))
    
    df_get['pretty_comp'] = df_get['comp_obj'].apply(lambda x: x.reduced_formula) #Hf3Zr2C3和Zr3Hf2C3的pretty comp不一样。一个是Hf3Zr2C3,另一个是Hf2(ZrC)3
    df_get['comp_dict'] = df_get['pretty_comp'].apply(lambda x: data_utils.parse_formula(x))
    
    # 'comp_dict': defaultdict(<class 'int'>, {'H': 1.0})
    df_get['comp_fractions'] = df_get['comp_dict'].apply(lambda x: data_utils.get_fractions(x)) 
    return df_get
    '''
    # 把comb0 ('Li', 'Be', 'Ti')变成{'Li':0.1,'Be':0.3,'Ti':0} 字典再变成dataframe。最终生成三元化合物的全元素组合dataframe。
    for comb0 in comb:
     t1 = time.time()
     print('combination',comb0) # ('Li', 'Be', 'Ti')
     for A,x,B,y,C,z in product('A',[x for x in np.round(np.arange(0,max_nelements,1),2)],   #[x for x in np.round(np.arange(0.1,1,0.1),2)] 生成0.1~0.9的list
                                'B',[x for x in np.round(np.arange(0,max_nelements,1),2)],
                                'C',[x for x in np.round(np.arange(0,max_nelements,1),2)]):  
     #print(x,y,z,f,e,g) 
     #print('add result',y+f+g)  
       if np.round(x+y+z,1)<=max_nelements: # 限制原子个数。#y+f+g==1,相当于限制分子中的原子个数为10个，去掉类似 A0.1B0.1C0.2， 注意了，x+f+g有有可能为1.00001，要去掉小数点。
         print(comb0[0],x,comb0[1],y,comb0[2],z) # 打印原子个数10以内的所有卡迪尔积组合
         dict0={comb0[0]:[x/(x+y+z)],comb0[1]:[y/(x+y+z)],comb0[2]:[z/(x+y+z)]} # 把AxByCz的系数变为小数点
         df1=pd.DataFrame(data=dict0)
         df=df.append(df1,ignore_index=True)
         df=df.fillna(0)  # 将NaN替换为0
         df.drop_duplicates(keep='first', inplace=True) # 除去重复系数。 如Li 0 Be 0 Ti 1和 Li 0 Be 0 Ti 2
    
     t2 = time.time()
     print('Generate',comb0,'combination time cost: %.6f'%(t2-t1)) # 生成('Li', 'Be', 'Ti')组合为5.787126 s
    df=df.drop([0]) #删除index=0的第一行，因为第一行全是0,0,0,0
    return df
    '''
#%
#df_combination=ternary_elements_product(elements_range=['Zr','C','Hf'],max_nelements=10) # test

#%% 生成原子数为10及以下的一元，二元，三元化合物全组合。
#df_combination=ternary_elements_product(elements_range=elements_in,max_nelements=10)#(elements_range=elements_in)
#%
#df_combination.to_csv('test00.csv')#('all_element_in_combination_max_nelements10.csv')
###df_combination=ternary_elements_product(elements_range=elements_combination,max_nelements=10)#(elements_range=elements_in,max_nelements=5)#(elements_range=elements_in)



#%% 3.2 Try a random forest model 重新拟合一次
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

y=df['K_VRH'].values
rf = RandomForestRegressor(n_estimators=50, random_state=1)

rf.fit(X, y)
print('training R2 = ' + str(round(rf.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=rf.predict(X))))
#%
prediction =rf.predict(df_combination[elements_in]) # 这predict之前要rf.fit(X,y)，不能单独用，cross_val_score（）也没用
#%
'''
#%% 画图 ternary三种元素formation_energy heatmap

# 构造三元图的点。
def ternary_heatmap(prediction,df_combination,elements_combination = ['Ta', 'W', 'Hf'],prop='K_VRH',dft_data_show=False):
    """
    Parameters
    ----------
    elements_combination : TYPE  list of ternry elements # 对应的[上，左，右]元素label。#这个不是随便取的！！！
        for generating df_combination

    df_combination : TYPE  DataFrame
        from ternary_elements_product function, for generating prediction
        
    prediction : TYPE # 1D arrayarray of float64  (250,) 
    # 不支持77个全元素组合为input的多输出multioutput，在输入前选择好prediction['K_VRH']或prediction['formation_energy']
        the prediction of bulk modulus, formation_energy from AI
    
    prop:性质 'K_VRH' or 'formation_energy' or 'e_above_hull' # 默认画k_VRH 
    
    dft_data:TYPE DataFrame from Material Project. df=pd.read_csv('../dataset/MaterialProject_input.csv')  # 86元素 装的是MP计算好的DFT property
    
    Returns
    -------
    None.

    """
    global dft_data
    if dft_data_show==True:
    # 装入 DFT property
#        
        dft_data=df[[x for x in elements_combination]] # 只装入elements_combination
        dft_data['property']=df[prop]  #装入 DFT property
    
#% 只要列'Ta', 'W', 'Hf'的相加为1的行
        dft_data = dft_data[np.isclose(np.round(dft_data[[x for x in elements_combination]].sum(axis=1),1), 1)] 
        print('dft_data\n',dft_data)
    #只要列'Ta', 'W', 'Hf'的相加为1的行
    #df_combination = prediction[np.isclose(np.round(prediction[[x for x in elements_combination]].sum(axis=1),1), 1)] 
    #print('prediction',prediction.shape) # 'Series' object has no attribute 'type'
    #print('df_combination',df_combination) #'DataFrame' object has no attribute 'type'
       
    #print(df_combination.columns)
    df_combination['property']=prediction # 如果df_combination已经有’enenrgy’这一列，则报错
    #print('property of df_combination',df_combination[['property']])
    import ternary
    import random
    import matplotlib
    import math
    import matplotlib.pyplot as plta
    import pandas as pd
    axes_colors = {'b': 'g', 'l': 'r', 'r':'b'} 
    
    import matplotlib
    matplotlib.rcParams['figure.dpi'] = 200
    #%
    scale = 10 #决定着格子数量(scale*scale)和刻度的scale等分，坐标范围
    # 从原始数据中，构造输入tax.heatmap的字典 # 转换 AxByCz的坐标为(z*scale,x*scale), x+y+z=1
    # AxByCz的坐标为(z*scale,x*scale),x+y+z=1 >>>> (x,y,z)>(z*scale,x*scale)
    # 坐标根据grid line 的左右刻度决定(right,left)，范围为(0~scale), (3,4,5)后面value=5没有数值范围。可以直接输入tax.heatmap
    '''
    df=pd.DataFrame({'A':[0.3,0.1],    # 元素+预测性能
                     'B':[0.6,0.1],
                     'C':[0.1,0.9],
                'property':[10,15]}) # top,left,right_corner_label 为A，B，C
    '''
    
    #('A','B','C')>(top,left,right)>(Ta,W,Hf)
    #point_x=(df_combination[['Hf']].values*scale).reshape(1,-1).tolist()[0]  #[['C']] # 不乘scale
    #point_y=(df_combination[['Ta']].values*scale).reshape(1,-1).tolist()[0] #[['A']]
   
    point_x=(df_combination[[elements_combination[2]]].values*scale).reshape(1,-1).tolist()[0]  #[['C']]对应'Hf' # 不乘scale
    point_y=(df_combination[[elements_combination[0]]].values*scale).reshape(1,-1).tolist()[0] #[['A']]对应'Ta'
    #%
    energy=df_combination[['property']].values.reshape(1,-1).tolist()[0]
    #%
    points=list(zip(point_x,point_y))
    
    
    #% 画heatmap需要把points放入points_values_dict字典中
    points_values_dict={}
    for i,point in enumerate(points):
        print(point)
        print(energy[i])
        dict1= {point:energy[i]}
        points_values_dict.update(dict1)
    
    if dft_data_show==True:
        point_x_dft=(dft_data[[elements_combination[2]]].values*scale).reshape(1,-1).tolist()[0]  #[['C']]
        point_y_dft=(dft_data[[elements_combination[0]]].values*scale).reshape(1,-1).tolist()[0] #[['A']]
        energy_dft=dft_data['property'].values.reshape(1,-1).tolist()[0]
        points_dft=list(zip(point_x_dft,point_y_dft))
    #%%    开始画图
    import ternary
    import random
    import matplotlib
    import math
    print("Version", ternary.__version__)
    figure, tax = ternary.figure(scale=scale) # scale 决定着热图中格子的多少，scale=3，则图中有3*3个格子
    
    # 热图数据导入
    # 画预测
    if prop=='formation_energy':
     tax.heatmap(points_values_dict, style="triangular",cbarlabel='Formation Energy (eV/atom)')#style="h"
    if prop=='K_VRH': 
     tax.heatmap(points_values_dict, style="triangular",cbarlabel='Bulk modulus (GPa)')#style="h"
    if prop=='e_above_hull': 
     tax.heatmap(points_values_dict, style="triangular",cbarlabel='Stability (eV/atom)')#style="h"
    else:
        tax.heatmap(points_values_dict, style="triangular",cbarlabel=prop)#style="h"
    
    # df_combination中max_nelements=100的点图，和heatmap图长差不多。所以预测还是用heatmap，max_nelements=10即可
    #tax.scatter(points,c=energy,vmax=max(energy),vmin=min(energy), marker='^', colorbar=False)
    if dft_data_show==True:
        tax.scatter(points_dft,c=energy_dft,vmax=max(energy),vmin=min(energy), marker='*', colorbar=False,label='DFT',s=200) #c=energy_dft代表color,vmin,vmax决定colorbar的值和点的颜色深浅,colormap=plt.cm.viridis
        tax.legend() # 要在scatter中写label才有 legend
    #tax.legend(a0,labels=['* DFT']) # 注释 legend(*args, **kwargs)
    
    #tax.colorbar.set_label('colorbar',fontdict=font)
    
    # add contour lines
    tax.contour(points_values_dict, levels=0.1, linewidths=2, colors='k', linestyles='solid', **kwargs)
    #levels,contour lines are drawn at equally spaced levels.
    
    
    tax.boundary() #三角形黑色边界
    # gridlines
    tax.gridlines(multiple=1, # 虚线的数量（scale的乘积）
                  linewidth=1,
                  horizontal_kwargs={'color':axes_colors['b']},
                  left_kwargs={'color':axes_colors['l']},
                  right_kwargs={'color':axes_colors['r']},
                  alpha=0.5,  ## 透明度 #RGBA values should be within 0-1 range 
                  ls='-') 
    #刻度
    ticks = [round(i / float(scale), 2) for i in range(scale+1)]  # 0~1.0,梯度0.1的list
    tax.ticks(axis='rlb', 
              linewidth=1,
              multiple=1, 
              clockwise=True, # True为刻度顺时针增长
              ticks=ticks,
              tick_formats="%0.1f", #ticks是刻度数值list #在三边上不显示小数点。 0.1显示为0, 要用formats 
              axes_colors=axes_colors, 
              fontsize=5,
              offset=0.015) # 刻度离三边的距离
    
    #tax.left_axis_label("$x_1$", offset=0.16, color=axes_colors['l'])
    #tax.right_axis_label("$x_0$", offset=0.16, color=axes_colors['r'])
    #tax.bottom_axis_label("$x_2$", offset=-0.06, color=axes_colors['b'])
    fontsize = 12
    # 三角形顶角坐标 ('A','B','C')>(top,left,right)>(Ta,W,Hf)
    tax.top_corner_label(elements_combination[0], fontsize=fontsize, offset=0.2,color=axes_colors['l']) #'A'>'Ta'
    tax.left_corner_label(elements_combination[1], fontsize=fontsize,color=axes_colors['b']) #‘B’>'W'
    tax.right_corner_label(elements_combination[2], fontsize=fontsize,color=axes_colors['r']) #'C'>'Hf'
    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off') # 删除方框刻度 #tax.ax.axis("off") # 删除方框刻度
    #tax.ax.axis("off")
    return figure
    #figure.set_facecolor('w') # 设置方框背景色，'w'是白，'r'是红， 默认就是白色
    
    #tax.set_title("Heatmap Test: Hexagonal")  #图正上方的标题
    
def ternary_contour_plotly(prediction,df_combination,elements_combination = ['Ta', 'W', 'Hf'],prop='K_VRH',dft_data_show=False):
    '''
    See https://plotly.com/python/ternary-contour/
    https://plotly.com/python-api-reference/generated/plotly.figure_factory.create_ternary_contour.html

    Parameters
    ----------
    prediction : TYPE np.array(200,)
        DESCRIPTION.
    df_combination : TYPE
        DESCRIPTION.
    elements_combination : TYPE, optional
        DESCRIPTION. The default is ['Ta', 'W', 'Hf'].
    prop : TYPE, optional
        DESCRIPTION. The default is 'K_VRH'.
    dft_data_show : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    '''
    import plotly.figure_factory as ff
    import numpy as np
    import plotly as plt
    Ta = np.array(df_combination[elements_combination[0]])
    W = np.array(df_combination[elements_combination[1]])
    Hf = np.array(df_combination[elements_combination[2]]) #1-Ta-W会出现负数5.5e-17，因为存储小数点
    #ValueError: The sum of coordinates should be 1 or 100 for all data points 说明Ta+W+Hf不是100%
    
    # synthetic data for mixing enthalpy
    # See https://pycalphad.org/docs/latest/examples/TernaryExamples.html
    
    fig = ff.create_ternary_contour(np.array([Ta,  Hf, W]), prediction,
                                    pole_labels=[elements_combination[0], elements_combination[2],elements_combination[1]],
                                    width=400,
                                    height=400,
                                    interp_mode='cartesian',
                                    ncontours=20,
                                    showscale=True,
                                    colorscale='Viridis',
                                    title=prop+ ' of ternary alloy',
                                    )
    
# 尝试Customize colorbar  没用
    #fig.update_layout(coloraxis_colorbar=dict(title='Intensity', 
                                               #tickvals=100#[0, 0.5, 1], 
                                               #ticktext=['Low', 'Medium', 'High']))
                                                         
    #fig.update_coloraxes(colorbar_dtick='L100')
    
    
    #fig.show()
    return fig
#%%
if __name__ == "__main__": # 当别的程序调用Ternary这个程序时候，这个命令以下的语句都不会执行。
#这在当你想要运行一些只有在将模块当做程序运行时而非当做模块引用时才执行的命令，只要将它们放到if __name__ == "__main__:"判断语句之后就可以了。
    import streamlit as st
    #figure=ternary_contour_plotly(prediction,df_combination,elements_combination = elements_combination,prop='Melting temperature (K)')#'formation_energy', 'Melting temperature (K)'
    #figure.show()
    #print('heatmap print sucessfully')


    
        #%%
#%放这个unit cell 在streamlit try会报错IndentationError: unindent does not match any outer indentation level    

    # Disable the display of errors in the web interface
    st.set_option('deprecation.showfileUploaderEncoding', True)
    try: #streamlit try:之后得代码都要在网页上运行，如model training
        
        '''
        unwanted_columns = ['energy', 'energy_per_atom', 'volume', 'formation_energy_per_atom',
               'nsites', 'unit_cell_formula', 'pretty_formula', 'elements',
               'nelements', 'e_above_hull','spacegroup', 'band_gap', 'density', 'cif', 'icsd_ids',
               'total_magnetization', 'elasticity', '_cell_length_a', '_cell_length_b',
               '_cell_length_c','_cell_angle_gamma', '_cell_angle_beta', '_cell_angle_alpha', 'G_VRH',
               'K_VRH', 'poisson_ratio', 'elastic_anisotropy', 'comp_obj',
               'pretty_comp', 'comp_dict', 'comp_fractions']
        X = df.drop(unwanted_columns, axis=1)
        print(X.columns)
        X=X.values
        '''

        
        
        option1 = st.selectbox(
        'Please pick up 1st element:',
        ('None','H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 
                   'K','Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se','Br',
                   'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',  'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In','Sn', 'Sb', 'Te', 'I',
                   'Cs', 'Ba', 
                   'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd','Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au','Hg', 'Tl', 'Pb', 'Bi')) #index=5 C  1
     
        
        option2 = st.selectbox(
        'Please pick up 2nd element:',
        ('None','H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 
                   'K','Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se','Br',
                   'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',  'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In','Sn', 'Sb', 'Te', 'I',
                   'Cs', 'Ba', 
                   'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd','Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au','Hg', 'Tl', 'Pb', 'Bi')) #index=36 Zr  2
    
        option3 = st.selectbox(
        'Please pick up 3rd element:',
        ('None','H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 
                   'K','Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se','Br',
                   'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',  'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In','Sn', 'Sb', 'Te', 'I',
                   'Cs', 'Ba', 
                   'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd','Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au','Hg', 'Tl', 'Pb', 'Bi')) #index 65 Hf   3
       
        #print('option1,2,3:',option1,option2,option3)
        #elements_combination = ['Ta', 'W', 'Hf']
        st.write('You selected:', option1,option2,option3)
#%
        elements_combination=[option1,option2,option3]
        
        #df_combination=ternary_elements_product(elements_range=elements_combination,max_nelements=10)
        #print('df_combination',df_combination)
        
        #%    #% sklearn Random forest model 训练和加载 
        elements_exclude = ['He','Ne','Ar','Kr','Tc','Xe','Pm','Po','At','Rn','Fr','Ra','Ac','Th', 'Pa', 'U', 'Np', 'Pu','Am', 'Cm', 
                            'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 
                            'Sg','Bh', 'Hs', 'Mt','Ds', 'Rg', 'Cn',]
        #print(elements_exclude) 
        elements_in = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 
                       'K','Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se','Br',
                       'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',  'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In','Sn', 'Sb', 'Te', 'I',
                       'Cs', 'Ba', 
                       'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd','Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                       'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au','Hg', 'Tl', 'Pb', 'Bi']
        
        print(len(elements_in)) #76
        
        #加载训练数据集
        df=pd.read_csv('../dataset/Merge prediciton100 and origin dataset_202403102024_03_17_00_59_02.csv') 
        #Merge prediciton100 and origin dataset_202403102024_03_17_00_59_02.csv' 高温+-50k,低温
        #Merge prediciton and origin dataset_202403102024_03_17_00_18_37.csv 这个高温物质太多了448128，低温3666种！低温预测不准确，如Li, Be, H
        #Merge prediciton and origin dataset_20240310.csv#这个高温物质太多了，导致连H都有3000多K！好像没有低温物质在里面。
        #('../dataset/MaterialProject_input.csv')  # 86元素
        
        feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                          cf.ValenceOrbital(props=['avg'])]) #, cf.IonProperty(fast=True)
        feature_labels = feature_calculators.feature_labels()
    
        X = df[elements_in+feature_labels]#df[elements_in]#df[elements_in+feature_labels] # 有重复的化合物
        #%
        #y=df['K_VRH'].values
        #y=df['formation_energy'].values
        y=df['Melting temperature (K)'].values
                #% sklearn Random forest model 训练和加载 
    
        #因为在streamlit try:命令下，所以streamlit run在网页上运行model training
        
        Trainable=False
        
        if Trainable==True:
            
            rf = RandomForestRegressor(n_estimators=200, criterion='mae',random_state=1) #n_estimators=50 criterion='mae' default mse，mae太慢了，
            rf.fit(X, y)
            print('training R2 = ' + str(round(rf.score(X, y), 3)))
            print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=rf.predict(X))))
            
            joblib.dump(rf, 'random_forest_regressor_Ternary1_20240311_v1'+datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S_")+'.pkl')
            
        else:
            rf = joblib.load('random_forest_regressor_Ternary1_20240311_v120240317_02_35_28_.pkl')
            #'random_forest_regressor_Ternary1_20240311_v120240317_02_35_28_.pkl' # n_estimator=100, mae, elementsin+3features as inputs
            #('20240317_00_25_00_random_forest_regressor_Ternary1_20240311_v1.pkl')
            #('random_forest_regressor_Ternary1_20240311_v1.pkl')
            print('Load model successfully')
    #% 
        #n_features=rf.（n_features_in_int）
    #%  test      
        #elements_combination=['C','Zr','Hf']#['C','Zr','Hf']#['H','Li','Be']#['C','Zr','Hf']
        from get_the_composition_featrures_from_formula20240310 import get_the_composition_featrures_from_formula
        df_combination=ternary_elements_product(elements_range=elements_combination,max_nelements=10)
        print('df_combination',df_combination)
        
        df_combination=get_the_composition_featrures_from_formula(df_combination)
        prediction =rf.predict(df_combination[elements_in+feature_labels]) # 这predict之前要rf.fit(X,y)，不能单独用，cross_val_score（）也没用
        
        df_combination['Melting temperature (K)']=prediction
        
        
    #% 把所有预测值看是否能替换成已有数据集中的数值。
        melting_point=pd.read_csv('../dataset/Merge prediciton and origin dataset_202403102024_03_17_00_18_37.csv')
        #%
        matched_df = pd.merge(melting_point,df_combination, on=elements_in) # on=列标签， 在列标签下寻找有相同值的行
        #%
        melting_point=[] #释放数据
        
        matched_df = matched_df.rename(columns={'comp_dict_x':'comp_dict',    'Melting temperature (K)_x': 'Melting temperature (K)'})
    #%    
       # 将'formula'列转换为字符串类型
        matched_df['comp_dict'] = matched_df['comp_dict'].astype(str) # TypeError: unhashable type: 'collections.defaultdict'  将'formula'列转换为字符串类型
        df_combination['comp_dict'] = df_combination['comp_dict'].astype(str)
        test=df_combination[['comp_dict','Melting temperature (K)']]
        
     
        test=pd.merge(df_combination,matched_df,how='left',on='comp_dict',suffixes=('_from_prediction', '_from_dataset')) #on=elements_in  #, suffixes=('_original', '_replacement') suffixes, 在两个dataframe中，如有相同列名，合并后的列名后缀。
    # TypeError: unhashable type: 'collections.defaultdict'  将'formula'列转换为字符串类型
    
    
    #%   将dataset中的值_from_dataset，替换成预测值_from_prediction，如果'Melting temperature (K)_from_dataset'有nan,则还是用预测值。
        test['melting temperature'] = test.apply(lambda row: row['Melting temperature (K)_from_dataset'] if pd.notna(row['Melting temperature (K)_from_dataset']) else row['Melting temperature (K)_from_prediction'], axis=1)
    #%  
        df_combination['Melting temperature (K)']=test['melting temperature']
        
        prediction=np.array(df_combination['Melting temperature (K)'])
    
    
    
    
    
        
       
        #%

        #figure=ternary_heatmap(prediction,df_combination,elements_combination = elements_combination,prop='formation_energy')
        #st.pyplot(figure) #用于展示ternary.figure的
        figure=ternary_contour_plotly(prediction,df_combination,elements_combination = elements_combination,prop='Melting temperature (K)')#'formation_energy', 'Melting temperature (K)'
        print('heatmap print sucessfully')
        st.plotly_chart(figure) #用于展示plotly的
        

