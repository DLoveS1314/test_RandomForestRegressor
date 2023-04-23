import pandas as pd
from factor_analyzer import FactorAnalyzer
# 计算巴特利特P值
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

def getdata():
    path = '/home/dls/data/openmmlab/test_RandomForestRegressor/large_samples_b60_all.csv'
    # path ='/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/FULLER4D_6_process.csv'
    # path ='/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/FULLER4H_6_process.csv'
    # path ='/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/FULLER4T_6_process.csv'
    # path = '/home/dls/data/openmmlab/test_RandomForestRegressor/merge_nopre_test.csv'
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_rows', 1000)
    data = pd.read_csv(path)
    data_y=['recalls',
    'recalls_norm',
    'accs',
    'accs_norm',
    'f1s',
    'f1s_norm',
    'precisions',
    'precisions_norm']
    x_name = [name for name in data.columns if name not in data_y]
    data_x =  data[x_name]
    data_y =  data[data_y]
    # print(data_x.head())
    # print(data_y.head())

    # need_name = ['area','per','zsc','disminmax','disavg','angleminmax','angleavg','csdminmax','csdavg','precisions' ]
    # data =data[need_name]
    # 将数据集分成训练集和测试集 最后一列是因变量 剩余的列是自变量
    # scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
    df_normalized = pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns)
    # df_normalized.head()

    corr = data_x.corrwith(data_y['f1s_norm'] )
    with open('/home/dls/data/openmmlab/test_RandomForestRegressor/log_x_norm.txt','w') as f :
        ind =corr.index.str.contains('_ir')
    
        f.write(
            f'Correlation between data_x and f1s_norm :\n{corr[~ind]}\n'
        )

    # 发现norm完后的结果普遍变好 利用spherephd处理的结果没有普通的好

    corr = df_normalized.corrwith(data_y['f1s_norm'] )
    with open('/home/dls/data/openmmlab/test_RandomForestRegressor/log_norm_norm.txt','w') as f :
        ind =corr.index.str.contains('_ir')
        out =corr[~ind]
        ind1 = out.index.str.contains('norm')
        out =out[ind1]
        f.write(
            f'Correlation between df_normalized and f1s_norm :\n{out}\n'
        )
    # 发现归一化与不归一化结果相同 但是归一化后 对随机森林方便验证

    #提取包含norm的值作为自变量

    namefilter =[name for name in df_normalized.columns if (('norm'in name) and ('_ir' not in name)) ]

    df_normalized = df_normalized[namefilter]##loc 去除方差行 选择多列和多行
    # newcolumns = []
    df_normalized_temp = pd.DataFrame()

    for name ,value in df_normalized.items():
        if 'std'in name :
            base=name.split('_')[0]
            newname = base+'_s'
            df_normalized_temp[newname] =value
        elif 'mean'in name :
            base=name.split('_')[0]
            newname = base+'_m'
            df_normalized_temp[newname] =value
        else:
            raise(f'name is {name}')
    df_normalized =df_normalized_temp
    df_normalized.columns
    return df_normalized
chi_square_value,p_value=calculate_bartlett_sphericity(LA_data_final_feat)
chi_square_value, p_value  
# 计算KMO值 
kmo_all,kmo_model=calculate_kmo(LA_data_final_feat)
print(kmo_model)
# 读取数据
data = pd.read_csv('data.csv')

# 初始化因子分析模型，并指定要提取的因子数量
fa = FactorAnalyzer(n_factors=3)

# 使用最大似然方法对数据进行因子分析
fa.fit(data)

# 输出因子载荷矩阵
loadings = fa.loadings_
print(loadings)