# 除以全格网的算术平均值 然后逐个文件进行归一化

from itertools import product
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import os
import json
import geopandas as gpd
from shapely.geometry import Point
# import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from main_new import dataprocessdata_rdp_nonorm
import matplotlib as mpl
from maketablefromfile import *
from shapely import geometry
# from factor_analyzer import FactorAnalyzer
# from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
# from factor_analyzer.factor_analyzer import calculate_kmo
# path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out'
# proj = ['FULLER','ISEA']
# topo = ['4D','4H','4T']
proj = ['FULLER' ]
topo = ['4D' ]
res = 5
# path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out'
# path = '/home/dls/data/openmmlab/mmclassification/tools/large_samples_b60/out'

json_path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out_ori'
if res==6:
    json_path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out_0_0'
# 处理json数据为df文件

def cnn_meanresult(path,kolds):
# 处理由mmcls生成的数据 对于交叉验证精度表格要进行取平均后再与几何属性进行合并

    for prj , tpo in product(proj,topo):
        dfs = []
        for k in kolds:
            df= pd.read_csv(os.path.join(path,f"{prj}{tpo}_{res}_{k}_cnnresults.csv"))
            # round_name = [name for name in df.columns if 'names' not in name]
            # 不需要那么多位数
            # df[round_name] = df[round_name].round(5)
            df =df.drop(['r_lat','r_lon','name'],axis=1)
            dfs.append(df)
        outpath = os.path.join(path,f"{prj}{tpo}_{res}_cnnmean.csv")
        df_con = pd.concat(dfs, axis=0, ignore_index=True)
        # meanname = [name for name in df_con.columns if name not in ['names','r_lon','r_lat']]
        # df_group = df_con.groupby([ 'names','r_lon','r_lat']) 
        df_group = df_con.groupby('seqnum') 
        print(f'groupby names {len(df_group.groups.keys())}')
        df_group_mean = df_group.mean()
        df_group_mean.reset_index(inplace=True)
        # df_group_mean = df_group_mean.rename(columns={'names': 'name'})
        df_group_mean.to_csv(outpath,index=False) 
        print(f'cnn_meanresult save result to {outpath}')
# 合并生成的统计量
def merge_geo_metric(outpath):
    # 如果改变了统计量 需要重新合并/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out 里的 calstatis 和 metrics 文件变为all文件 与savepad功能相同
    # path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out'
    # proj = ['FULLER','ISEA']
    # topo = ['4D','4H','4T']
    for prj , tpo in product(proj,topo):
        if res ==6:
            dirname_calstatis = os.path.join(outpath,f"{prj}{tpo}_{res}_calstatis.csv")
            dirname_metircs = os.path.join(outpath,f"{prj}{tpo}_{res}_metrics.csv")
            
            dirname_all = os.path.join(outpath,f"{prj}{tpo}_{res}_cnngeo.csv")
            df_cal =pd.read_csv(dirname_calstatis)
            # print(df_cal.shape)
            df_met =pd.read_csv(dirname_metircs)
            # print(df_met.shape)
            merged_df = pd.merge(df_cal, df_met, on=['name'], how='inner')
            # print(merged_df.columns)
            # print(f'save result to {dirname_all}')
            # print(merged_df.shape)
            merged_df.to_csv(dirname_all,index=False)
        else:
            dirname_metircs = os.path.join(outpath,f"{prj}{tpo}_{res}_cnnmean.csv")
            df_met =pd.read_csv(dirname_metircs)

            dirname_calstatis = os.path.join(outpath,f"{prj}{tpo}_{res}_geostatis.csv")
            df_cal =pd.read_csv(dirname_calstatis)
            df_cal= df_cal.drop([ 'r_lon', 'r_lat' ,'name','hasdata'], axis=1 )
 
            
            merged_df = pd.merge(df_cal, df_met, on=['seqnum'], how='inner')

            dirname_all = os.path.join(outpath,f"{prj}{tpo}_{res}_cnngeo.csv")
            merged_df.to_csv(dirname_all,index=False)

# 对于非RDP的数据 需要计算固定区域的cnn结果

def getoriscore(json_dirname):
    with open(json_dirname, 'r') as file:
        json_str = file.read()
    monitor ={'accs': 'accuracy/top1','precisions':'single-label/precision','recalls':'single-label/recall','f1s':'single-label/f1-score' }
    # 将JSON转换为字典
    jsondata = json.loads(json_str)
    # for key,value in monitor.items():
    #     current_score = jsondata[value]  
    dictval ={ key:[jsondata[value]] for key,value in monitor.items() }
    # 打印字典
    # print(dictval)
    dftestval =pd.DataFrame.from_dict(dictval)
    return dftestval
# 处理生成的json 把 固定区域的test值 进行三个平均
def cnn_meanresultori(oripath):
    for prj , tpo in product(proj,topo):
        if res ==6:
            json_dirname = os.path.join(oripath,f"{prj}{tpo}_l{res}_00.json")
            json_dirname=json_dirname.lower()
            # 与res5后续的程序保持一直
            df_con_mean = getoriscore(json_dirname)#0 0 位置的预测精度
         
        else:
            dfs = []
            for k in ['0','1','2']:
                name =  f'resnet18_ucmf1_{prj}{tpo}_l{res}_{k}.json'.lower ()
                dirname = os.path.join(oripath,name)
                df = getoriscore(dirname)
                dfs.append(df)
            # 只有三行 无需再group
            df_con = pd.concat(dfs, axis=0, ignore_index=True)
            df_con_mean = df_con.mean()
        #   把serise转化为df并把行转化为列
        # df_T =pd.DataFrame()
        #     # print(df_con_mean.index)
        # for name in df_con_mean.index:
        #     # print(name,np.array(df_con_mean.loc[name]))
        #     df_T[str(name)] =[df_con_mean.loc[name]] 
        outpath = os.path.join(oripath,f"{prj}{tpo}_{res}_ori_cnnmean.csv")
        df_con_mean.to_csv(outpath ,index =False)
        # my_df = pd.read_csv(outpath, header=0, skiprows=[0])
        # print(df_T )
        # my_df.to_csv(outpath ,index =False)
        
        
# 处理 由mmcls生成的数据（ 包含几何属性 和 预测精度 ）删除不需要的属性 并进行归一化
def processdata(outpath,cnngeopath):
    for prj , tpo in product(proj,topo):
        if res ==  6:
            dirname = os.path.join(cnngeopath,f"{prj}{tpo}_{res}_cnngeo.csv")
            df =pd.read_csv(dirname)
            # 筛选包含 mean 或 std 的列 默认filter列筛选 想行筛选 坐标轴选择 axis=0
            columns_with_mean = df.filter(like='_mean').columns
            columns_with_std = df.filter(like='_std').columns
            for column_name, column_data in df[columns_with_mean].items ():#遍历每一列
                stat_name = column_name.split('_')[0]  # 获取统计名称，如 area
                std_column_name = f'{stat_name}_std'  # 构造对应的 std 列名，如 area_std
                if std_column_name in columns_with_std:#如果有对应std名
                    std_column_data = df[std_column_name]  # 获取对应的 std 列数据
                    df[f'{stat_name}_cv'] = std_column_data / column_data     # 计算 mean/std 的值
            #删除多余的变量
            # df_drop= df_sorted.drop(['name', 'r_lon', 'r_lat','recalls','accs','f1s'], axis=1 )
            df_drop= df.drop(['name', 'r_lon', 'r_lat' ], axis=1 )
            drop_data = df[['r_lon', 'r_lat' ]]  
            # df_drop= df
            ori_cnn_dir= os.path.join(json_path,f"{prj}{tpo}_{res}_ori_cnnmean.csv")
            # json_dirname=json_dirname.lower()
            ori_cnn_result = pd.read_csv(ori_cnn_dir)
            
            csv_dirname = os.path.join(json_path,"out",f"{prj}{tpo}_{res}_00_calstatis.csv") #0 0 位置处的格网几何属性
            ori_geo_result = pd.read_csv(csv_dirname)
            ori_geo_result= ori_geo_result.drop(['name', 'r_lon', 'r_lat' ], axis=1 )
 
            mean_geo_dir= '/home/dls/data/openmmlab/test_RandomForestRegressor/mean'
            mean_name = os.path.join(mean_geo_dir,f"{prj}{tpo}_{res}_mean.csv")
            dggrid_mean_geo =  pd.read_csv(mean_name)
            
            for key ,val in dggrid_mean_geo.items():##使用变化率进行标准化
                value = float(val) 
                # 没有负值 用比率来表示更好 变化率无法显示变大还是变小
                df_drop[f'{key}_meannorm'] = df_drop[key] /value
                df_drop[f'{key}_meanratio'] = abs( (value-df_drop[key])/value ) ##几
                
            for key ,val in ori_geo_result.items():##使用变化率进行标准化
                value = float(val) 
                # 没有负值 用比率来表示更好 变化率无法显示变大还是变小
            
                df_drop[f'{key}_norm'] = df_drop[key] /value
                df_drop[f'{key}_ratio'] = abs( (value-df_drop[key])/value ) ##几何属性数据标准化一下 存成对应的norm数据 利用的是 0 0 位置处进行标准化
            
            print(f'ori_cnn_result {ori_cnn_result}')
            
            for key ,val in ori_cnn_result.items():
                print(key ,value)
                
                value = float(val) 
                df_drop[f'{key}_norm']  =df_drop[key] /value
                df_drop[f'{key}_ratio'] = abs( (value-df_drop[key])/value )##预测精度数据标准化一下 存成对应的norm数据 利用的是 0 0 位置处进行标准化
            
            # 对自变量进行均值的归一化 其实不合理
            # csv_mean = os.path.join('/home/dls/data/openmmlab/test_RandomForestRegressor/mean',f"{prj}{tpo}_{res}_mean.csv") #0 0 位置处的格网几何属性
            # metric_mean = pd.read_csv(csv_mean)
            # for key ,val in metric_mean.items():##因为涉及负值 所以用变化率来表示 abs((a-b)/a)
            #     value = float(metric_mean[key]) 
            #     df_drop[f'{key}_meannorm'] = abs( (value-df_drop[key])/value ) ##用算术平均值norm存成对应的norm数据 利用的是全球位置的均值进行标准化
            #     # 对数据进行归一化 本身就是逐列归一化
        
            # 交叉验证上次的结果已经不包含这些列 另外又包含了 从c++处就生成的 maxmin为了避免误删这里注释掉了
            df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_var')]##loc 去除方差行 选择多列和多行
            df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_cv')]##loc 去除方差行 选择多列和多行
            df_drop = df_drop.loc[:,~df_drop.columns.str.contains('max')]##loc 去除方差行 选择多列和多行
            df_drop = df_drop.loc[:,~df_drop.columns.str.contains('min')]##loc 去除方差行 选择多列和多行
            df_drop = df_drop.loc[:,~df_drop.columns.str.contains('maxmin')]##loc 去除方差行 选择多列和多行
            
            df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_ir')]##loc 去除方差行 选择多列和多行
            # 只选择带norm的列
            # df_drop = df_drop.loc[:,df_drop.columns.str.contains('norm')]##loc 去除方差行 选择多列和多行       
            scaler = MinMaxScaler()
            df_drop = df_drop[sorted(df_drop.columns)]
            # 把 name r_lon r_lat 加上 为了后续出图用
            df_drop=pd.concat([df_drop,drop_data],axis=1)
            os.makedirs(outpath,exist_ok=True)
            outdirname = os.path.join(outpath,f"{prj}{tpo}_{res}_process_noscaler.csv")
            df_drop.to_csv(outdirname,index=False)
            
            outdirname = os.path.join(outpath,f"{prj}{tpo}_{res}_process_scaler.csv")
            df_drop_sacler= pd.DataFrame(scaler.fit_transform(df_drop ), columns=df_drop.columns)
            df_drop_sacler.to_csv(outdirname,index=False)
    
            print(f'dirname: {dirname} outdirname :{outdirname}')
        else :
            dirname = os.path.join(cnngeopath,f"{prj}{tpo}_{res}_cnngeo.csv")
            df =pd.read_csv(dirname)
            # df_drop= df_sorted.drop(['name', 'r_lon', 'r_lat','recalls','accs','f1s'], axis=1 )
            df_drop= df.drop(['seqnum','name', 'r_lon', 'r_lat' ], axis=1 )
            drop_data = df['seqnum']  
            # df_drop= df
            ori_cnn_dir= os.path.join(json_path,f"{prj}{tpo}_{res}_ori_cnnmean.csv")
            # json_dirname=json_dirname.lower()
            ori_cnn_result = pd.read_csv(ori_cnn_dir)
            ori_geo_dirname = os.path.join(json_path, f"{prj}{tpo}_{res}_ori_geostatis.csv") #0 0 位置处的格网几何属性
            ori_geo_result = pd.read_csv(ori_geo_dirname)
            ori_geo_result= ori_geo_result.drop(['name', 'r_lon', 'r_lat' ], axis=1 )
            
            
            mean_geo_dir= '/home/dls/data/openmmlab/test_RandomForestRegressor/mean'
            mean_name = os.path.join(mean_geo_dir,f"{prj}{tpo}_{res}_mean.csv")
            dggrid_mean_geo =  pd.read_csv(mean_name)
            
            for key ,val in dggrid_mean_geo.items():##使用变化率进行标准化
                value = float(val) 
                # 没有负值 用比率来表示更好 变化率无法显示变大还是变小
                df_drop[f'{key}_meannorm'] = df_drop[key] /value
                df_drop[f'{key}_meanratio'] = abs( (value-df_drop[key])/value ) ##几
                
            for key ,val in ori_geo_result.items():##使用变化率进行标准化
                value = float(val) 
                # 没有负值 用比率来表示更好 变化率无法显示变大还是变小
                df_drop[f'{key}_norm'] = df_drop[key] /value
                df_drop[f'{key}_ratio'] = abs( (value-df_drop[key])/value ) ##几何属性数据标准化一下 存成对应的norm数据 利用的是 0 0 位置处进行标准化
            for key ,val in ori_cnn_result.items():
                value = float(val) 
                # print(key ,value)
                df_drop[f'{key}_norm']  =df_drop[key] /value
                df_drop[f'{key}_ratio'] = abs( (value-df_drop[key])/value )##预测精度数据标准化一下 存成对应的norm数据 利用的是 0 0 位置处进行标准化
            
            # 对自变量进行均值的归一化 其实不合理
            # csv_mean = os.path.join('/home/dls/data/openmmlab/test_RandomForestRegressor/mean',f"{prj}{tpo}_{res}_mean.csv") #0 0 位置处的格网几何属性
            # metric_mean = pd.read_csv(csv_mean)
            # for key ,val in metric_mean.items():##因为涉及负值 所以用变化率来表示 abs((a-b)/a)
            #     value = float(metric_mean[key]) 
            #     df_drop[f'{key}_meannorm'] = abs( (value-df_drop[key])/value ) ##用算术平均值norm存成对应的norm数据 利用的是全球位置的均值进行标准化
            #     # 对数据进行归一化 本身就是逐列归一化
        
            # 交叉验证上次的结果已经不包含这些列 另外又包含了 从c++处就生成的 maxmin为了避免误删这里注释掉了
            # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_var')]##loc 去除方差行 选择多列和多行
            # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_cv')]##loc 去除方差行 选择多列和多行
            # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('max')]##loc 去除方差行 选择多列和多行
            # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('min')]##loc 去除方差行 选择多列和多行
            # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('maxmin')]##loc 去除方差行 选择多列和多行
            
            # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_ir')]##loc 去除方差行 选择多列和多行
            #只选择带norm的列
            # df_drop = df_drop.loc[:,df_drop.columns.str.contains('norm')]##loc 去除方差行 选择多列和多行       
            scaler = MinMaxScaler()
            df_drop = df_drop[sorted(df_drop.columns)]
            # 把 name r_lon r_lat 加上 为了后续出图用
            df_drop=pd.concat([df_drop,drop_data],axis=1)
            os.makedirs(outpath,exist_ok=True)
            outdirname = os.path.join(outpath,f"{prj}{tpo}_{res}_process_noscaler.csv")
            df_drop.to_csv(outdirname,index=False)
            
            outdirname = os.path.join(outpath,f"{prj}{tpo}_{res}_process_scaler.csv")
            df_drop_sacler= pd.DataFrame(scaler.fit_transform(df_drop ), columns=df_drop.columns)
            df_drop_sacler.to_csv(outdirname,index=False)
    
            print(f'dirname: {dirname} outdirname :{outdirname}')
def calmean(geopath,calmeanoutpath):
    outpath= calmeanoutpath
    path = geopath
    for prj , tpo in product(proj,topo):
        dirname = os.path.join(path,f"{prj}{tpo}_{res}_all.csv")
        outname = os.path.join(outpath,f"{prj}{tpo}_{res}_mean.csv")
        df =pd.read_csv(dirname)
        # break
        df.drop( columns=['seqnum'],inplace=True)
        means = df.mean()
        cv =  df.std() / df.mean()##避免均值为0
        stds = df.std()
        # print(df.columns)
        
        # print(dfout.head())
        # 创建一个新的DataFrame，其中每列的名称为以前列名加上相应的统计量
        new_df = pd.DataFrame(columns=[col + '_' + stat for col in df.columns for stat in ['mean', 'cv', 'std' ]])
        # if df_gmean:
        #     for col in df.columns:
        #         new_df[col + '_ir'] = np.log(df[col]/df_gmean[col + '_gmean'])
        # 将每个统计量添加到新的DataFrame中
        for col in df.columns:
            new_df[col + '_mean'] = [means[col]]
            new_df[col + '_cv'] = [cv[col]]
            new_df[col + '_std'] = [stds[col]]
        new_df.to_csv(outname,index=False)
# rdp 没有固定区域的cnn结果
def processdata_rdp(outpath,cnngeopath,outdirname) :
    for prj , tpo in product(proj,topo):
        dirname = os.path.join(cnngeopath,f"{prj}{tpo}_{res}_cnngeo.csv")
        df =pd.read_csv(dirname)
        # df_drop= df_sorted.drop(['name', 'r_lon', 'r_lat','recalls','accs','f1s'], axis=1 )
        df_drop= df.drop(['seqnum'], axis=1 )
        drop_data = df['seqnum']  
        
        mean_geo_dir= '/home/dls/data/openmmlab/test_RandomForestRegressor/mean'
        mean_name = os.path.join(mean_geo_dir,f"{prj}{tpo}_{res}_mean.csv")
        dggrid_mean_geo =  pd.read_csv(mean_name)
        df_meannorm = pd.DataFrame()
        for key ,val in dggrid_mean_geo.items():##使用变化率进行标准化
            value = float(val) 
            # 没有负值 用比率来表示更好 变化率无法显示变大还是变小
            df_meannorm[f'{key}_meannorm'] = df_drop[key] /value
            df_meannorm[f'{key}_meanratio'] =  (df_drop[key]-value)/value   ##几

            

        maxscaler = MinMaxScaler()
        meanscaler = StandardScaler()
        df_drop_maxsacler= pd.DataFrame(maxscaler.fit_transform(df_drop ), columns=[f'{column}_maxscaler'for column in df_drop.columns ])
        df_drop_meansacler= pd.DataFrame(meanscaler.fit_transform(df_drop ), columns=[f'{column}_meanscaler'for column in df_drop.columns ])
        
        df_all = pd.concat([drop_data,df_drop,df_meannorm,df_drop_maxsacler,df_drop_meansacler],axis=1)
        df_all = df_all[sorted(df_all.columns)]
        df_all_split= df_all.loc[df_all['seqnum'].isin(list(range(1,1025)))]
        outdirname = os.path.join(outpath,outdirname)
        df_all_split.to_csv(outdirname,index=False)
        print(f'dirname: {dirname} outdirname :{outdirname}')

        # 对自变量进行均值的归一化 其实不合理
        # csv_mean = os.path.join('/home/dls/data/openmmlab/test_RandomForestRegressor/mean',f"{prj}{tpo}_{res}_mean.csv") #0 0 位置处的格网几何属性
        # metric_mean = pd.read_csv(csv_mean)
        # for key ,val in metric_mean.items():##因为涉及负值 所以用变化率来表示 abs((a-b)/a)
        #     value = float(metric_mean[key]) 
        #     df_drop[f'{key}_meannorm'] = abs( (value-df_drop[key])/value ) ##用算术平均值norm存成对应的norm数据 利用的是全球位置的均值进行标准化
        #     # 对数据进行归一化 本身就是逐列归一化
    
        # 交叉验证上次的结果已经不包含这些列 另外又包含了 从c++处就生成的 maxmin为了避免误删这里注释掉了
        # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_var')]##loc 去除方差行 选择多列和多行
        # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_cv')]##loc 去除方差行 选择多列和多行
        # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('max')]##loc 去除方差行 选择多列和多行
        # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('min')]##loc 去除方差行 选择多列和多行
        # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('maxmin')]##loc 去除方差行 选择多列和多行
        
        # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_ir')]##loc 去除方差行 选择多列和多行
        #只选择带norm的列
        # df_drop = df_drop.loc[:,df_drop.columns.str.contains('norm')]##loc 去除方差行 选择多列和多行       
        # scaler = MinMaxScaler()
        # df_drop = df_drop[sorted(df_drop.columns)]
        # # 把 name r_lon r_lat 加上 为了后续出图用
        # df_drop=pd.concat([df_drop,drop_data],axis=1)
        # os.makedirs(outpath,exist_ok=True)
        # outdirname = os.path.join(outpath,f"{prj}{tpo}_{res}_process_noscaler.csv")
        # df_drop.to_csv(outdirname,index=False)
        
        # outdirname = os.path.join(outpath,f"{prj}{tpo}_{res}_process_scaler.csv")
        # df_drop_sacler= pd.DataFrame(scaler.fit_transform(df_drop ), columns=df_drop.columns)
        # df_drop_sacler.to_csv(outdirname,index=False)

        # print(f'dirname: {dirname} outdirname :{outdirname}')

# 把processdata 处理好的文件进行合并 用于随机森林训练
def mergeprocessdata(filepath ):
    # 读取文件夹中所有 CSV 文件
    merged_noscaler =[]
    merged_scaler =[]
    for prj , tpo in product(proj,topo):
        dirname_nosacler = os.path.join(filepath,f"{prj}{tpo}_{res}_process_noscaler.csv")
        df =pd.read_csv(dirname_nosacler)
        # 读取 CSV 文件并将数据合并到 DataFrame 中
        df = pd.read_csv(dirname_nosacler)
        merged_noscaler.append(df)
        
        dirname_scaler = os.path.join(filepath,f"{prj}{tpo}_{res}_process_scaler.csv")
        df =pd.read_csv(dirname_scaler)
        # 读取 CSV 文件并将数据合并到 DataFrame 中
        df = pd.read_csv(dirname_scaler)
        merged_scaler.append(df)
    df_ns = pd.concat(merged_noscaler)
    df_ns = df_ns.to_csv(os.path.join(filepath,f"merge_process_noscaler_{res}.csv"),index=False)
    df_s = pd.concat(merged_scaler)
    df_s = df_s.to_csv(os.path.join(filepath,f"merge_process_scaler_{res}.csv"),index=False)

# 计算指标之间的相关性
def calrel(path,outpath,dirname):
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_rows', 1000)
    data = pd.read_csv(path)
    # data =data.drop(['r_lat','r_lon'],axis=1)
    data =data.drop(['seqnum'],axis=1)

    data_y_name =[]
    for name in[ 'recalls',    'accs',    'f1s',    'precisions']:
         name0 =f'{name}_maxscaler'
         name1 =f'{name}_meanscaler'
         name2 =f'{name}'
         data_y_name.append(name0)
         data_y_name.append(name1)
         data_y_name.append(name2)
         
    # scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
    # df_normalized= pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    # x_name = [name for name in data.columns if (name not in data_y_name) and ('_cv' not in name) and ('_meanratio' in name) ]
    x_name = [name for name in data.columns if (name not in data_y_name) and ('_cv' not in name) and ('_std_std' not in name) and ('_std_mean' not in name)   ]

    data_x =  data[x_name]
    data_y =  data[data_y_name]
    # 只读取包含maxscaler的列
    data_x =  data_x.loc[:,data_x.columns.str.contains('_maxscaler')]
    # data_x_scaler =  df_normalized[x_name]
    # data_y_scaler =  df_normalized[data_y_name]
    # need_name = ['area','per','zsc','disminmax','disavg','angleminmax','angleavg','csdminmax','csdavg','precisions' ]
    # data =data[need_name]
    # 将数据集分成训练集和测试集 最后一列是因变量 剩余的列是自变量
    # scaler.fit(data_x)
    # scaler.fit(data_y)
    # df_normalized.head()
 
    # outpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/outcalrea'
    os.makedirs(outpath,exist_ok=True)
    
    coors =pd.DataFrame()
    for y_name in ['f1s_maxscaler' ] :
        corr0 = data_x.corrwith(data_y[y_name] )
        coors[f'x_y_{y_name}'] =corr0
        # corr1 = data_x.corrwith(data_y_scaler[y_name] )
        # coors[f'x_ysc_{y_name}'] =corr1
        
        # corr2 = data_x_scaler.corrwith(data_y_scaler[y_name] )
        # coors[f'xsc_ysc_{y_name}'] =corr2
        
        # corr3 = data_x_scaler.corrwith(data_y[y_name] )
        # coors[f'xsc_y_{y_name}'] =corr3
    print(coors.to_csv(os.path.join(outpath,dirname ))) 

    df_concat = pd.concat([data_x,data_y['f1s_maxscaler']],axis=1)
    df_concat.to_csv(os.path.join(outpath,'calrelout_withstd.csv'),index=False)
    # corr1 = data_x.corrwith(data_y_scaler[['f1s_ratio','f1s','f1s_norm']] )
    # corr1.to_csv(os.path.join(path,'x_yscaler.csv'))
    
    # corr2 = data_x_scaler.corrwith(data_y_scaler[['f1s_ratio','f1s','f1s_norm']] )
    
    # corr2.to_csv(os.path.join(path,'xscaler_yscaler.csv'))
    
    # corr3 = data_x_scaler.corrwith(data_y[['f1s_ratio','f1s','f1s_norm']] )
    # corr3.to_csv(os.path.join(path,'xscaler_y.csv'))
    
    
    
    # corr = data_x.corrwith(data_y['f1s_ratio'] )
    # print(corr)
    # corr.to_csv(os.path.join(path,'x_y.csv'))
    
    # corr = data_x.corrwith(data_y['f1s_ratio'] )
   
    # print(
    #         f'Correlation between data_x and f1s_ratio :\n{corr }\n'
    #     )
    # # 发现norm完后的结果普遍变好 利用spherephd处理的结果没有普通的好

    # corr1 = data_x_scaler.corrwith(data_y_scaler['f1s_ratio'] )
    # out_df['data_x_scaler_f1s_ratio']=corr1
    # print(
    #     f'Correlation between df_normalized and f1s_ratio :\n{corr1 }\n'
    # )
    # corr = data_x.corrwith(data_y['f1s_norm'] )
    # out_df['data_x_scaler_data_y_scaler']=corr1
    # print(
    #         f'Correlation between data_x and f1s_norm :\n{corr }\n'
    #     )
    # # 发现norm完后的结果普遍变好 利用spherephd处理的结果没有普通的好

    # corr1 = data_x_scaler.corrwith(data_y_scaler['f1s'] )
    # print(
    #     f'Correlation between df_normalized and f1s_norm :\n{corr1 }\n'
    # )
    # corr = data_x.corrwith(data_y['f1s'] )
    # print(
    #         f'Correlation between data_x and f1s :\n{corr }\n'
    #     )
    # # 发现norm完后的结果普遍变好 利用spherephd处理的结果没有普通的好

    # corr1 = data_x_scaler.corrwith(data_y_scaler['f1s'] )
    # print(
    #     f'Correlation between df_normalized and f1s :\n{corr1 }\n'
    # )

# 使用dataprocessdata_rdp_nonorm 得出相关性大的列 进行保存 并返回相关性大的数据，方便后续绘图
def calrel2(processname,outname=None ):
    dataout =dataprocessdata_rdp_nonorm(processname=processname)
    namex = dataout.columns[:-1]
    namey = dataout.columns[-1]
 
    datax = dataout[namex]
    datay = dataout[namey]
    print(f'namex {namex}')
    print(f'namey {namey}')
    corr0 = datax.corrwith(datay  )
    if outname is not None:
        dirpath = os.path.dirname(outname)
        corr0.to_csv(os.path.join(dirpath,'correl.csv' ))
        dataout.to_csv(outname,index=False)
    print(f'corr0 {corr0}') 
    df_all = pd.concat([datax,datay],axis=1)
    return  df_all
# 筛选掉相关性小的列
def getcolumnsbycorrel(processname):

    dirname = processname
    df =pd.read_csv(dirname)
    df_split= df.loc[df['seqnum'].isin(list(range(1,1025)))]
    drop_data =   df_split.drop(['seqnum' ,'recalls','precisions','f1s'], axis=1 )
    # 删除不需要的列 相关性小于一定值的列进行排除
    data_y_name =[ 'accs']
    # 处理x
    x_name = [name for name in drop_data.columns if (name not in data_y_name)  ]
    data_x =  drop_data[x_name]
    # scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
    # df_MinMax= pd.DataFrame(scaler.fit_transform(df ), columns=df.columns)
    # scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
    # x 归一化
    # data_x= pd.DataFrame(scaler.fit_transform(data_x ), columns=data_x.columns)

    data_x =  data_x.loc[:,~data_x.columns.str.contains('_cv')]
    data_x =  data_x.loc[:,~data_x.columns.str.contains('_area')]
    data_x =  data_x.loc[:,~data_x.columns.str.contains('per')]
    data_x =  data_x.loc[:,~data_x.columns.str.contains('frac')]
    data_x =  data_x.loc[:,~data_x.columns.str.contains('_std_std')]
    data_x =  data_x.loc[:,~data_x.columns.str.contains('_std_mean')]
    data_x =  data_x.loc[:,~data_x.columns.str.contains('edge_dis')]

    # y不处理
    data_y =  drop_data[data_y_name]
    
    print('data_x',data_x.columns)
    print('data_y',data_y.columns )
    corr0 = data_x.corrwith(data_y['accs']  )
    print(corr0[abs(corr0)>0.4])
    columnsx = list(corr0.index[abs(corr0)>0.4])
    columns =columnsx+['accs']
    print(columns)
    return  columns
def factor_ana(filename):
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_rows', 1000)
    abs_path = os.path.dirname(filename)
    df = pd.read_csv(filename)
    # 只取一个面的来做因子分析 
    # NOTE 只计算菱形的
    num =10 
    len_oneface = int(df.shape[0]/num)
    print(f'factor_ana len_oneface {len_oneface} ')
    seqnums = range(1,len_oneface+1)
    df_split =df[df['seqnum'].isin(seqnums)]
    df_split =df_split.drop(['seqnum'],axis=1)

    maxscaler = MinMaxScaler()
    meanscaler = StandardScaler()

    df_maxsacler= pd.DataFrame(maxscaler.fit_transform(df_split ), columns=[f'{column}_maxscaler'for column in df_split.columns ])
    df_meanscaler= pd.DataFrame(meanscaler.fit_transform(df_split ), columns=[f'{column}_meanscaler'for column in df_split.columns ])

 
    df_maxsacler.to_csv(os.path.join(abs_path,f'{os.path.basename(filename)}_maxsacler.csv'),index=False)
    df_meanscaler.to_csv(os.path.join(abs_path,f'{os.path.basename(filename)}_meanscaler.csv'),index=False)
    # df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_var')]##loc 去除方差行 选择多列和多行

    df_filter = df_maxsacler.loc[:,~df_maxsacler.columns.str.contains('_std')]
    
    chi_square_value,p_value=calculate_bartlett_sphericity(df_filter)
    print('bartlett',chi_square_value, p_value ) 
    # 计算KMO值 
    kmo_all,kmo_model=calculate_kmo(df_filter)
    print('kmo_model',kmo_model)
    fa = FactorAnalyzer(  n_factors=6, rotation='varimax')
    fa.fit(df_filter)
    # 输出因子载荷矩阵
    loadings = fa.loadings_
    loadings[np.where(loadings<0.4)]=0.0
    loadings =np.round(loadings,3)
    loading_df =pd.DataFrame(loadings,index= df_filter.columns)
    print(loading_df)
    factor_variance = fa.get_factor_variance()
    print(f'累计方差矩阵：\n{factor_variance}')
    # 得到特征值ev、特征向量v
    ev,_=fa.get_eigenvalues()
    print(f'特征值：\n{ev}')
# 绘图程序
def get_geopandas(path,titlename=None):
    # path = '/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/large_samples_b60_scaler/FULLER4D_6_process_sacler.csv'
    name = os.path.basename(path)
    df = pd.read_csv(path)
    # scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
    print(sum(df['f1s_norm']>0.4))
    # print(df.columns)
    # geometry = gpd.points_from_xy(df.r_lon, df.r_lat, crs="EPSG:4326")
    # geometry = [Point(xy) for xy in zip(df.r_lon, df.r_lat)]
    # # 创建几何对象列
    # # 创建Geopandas数据框
    # gdf = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=geometry)
    # # print(gdf.head()) 
    # ax = gdf.plot(column='f1s_norm', cmap='OrRd', legend=True)
    # ax.set_title(titlename)
    # ax.figure.savefig(f'{name}.png')

# 转化为xarray 画图 方便的是可以直接进行经纬度的画图 用于画f1score比较好
def get_xarray(path,titlename=None):
    crs = ccrs.Orthographic(0.0,0.0 )
    crs1 =ccrs.Robinson(0.0)
    ax = plt.subplot( projection=crs )
    # fig, ax = plt.subplots(figsize=(4,8))
    name = os.path.basename(path)
    print('name',path)
    df = pd.read_csv(path)
    df = df.set_index(['r_lat','r_lon'])
    ds = xr.Dataset.from_dataframe(df  )
    p = ds['f1s'].plot( transform=ccrs.PlateCarree(),ax=ax)
    # p = ds['f1s_norm'].plot( ax=ax)
    
    # p.axes.set_global()
    ax.set_title(titlename)
    p.axes.gridlines()
    p.axes.coastlines()
    # p.axes.set_extent([120, 250, -75, 75], crs=ccrs.PlateCarree())
    draw_dggrid(p.axes,2)
    # draw_dggrid(p.axes,0)

    
    p.figure.savefig(f'{name}_xr.png')
# 目的是删除一些不与line相交的格网 这各函数是辅助函数 返回相交的格网有哪些
def check_intersects(geometry,line):
    return geometry.apply(lambda geom: geom.intersects(line))

def draw_dggrid(dggtype,res,lon,lat,ax):    #  创建绘制的格网
    gdf= creatcellandcode(dggtype,res,dens='40')
    gdf = gdf.set_crs('EPSG:4326') 
    gdf= gdf.set_geometry('cell')
    gdf.sort_values('seqnum',ascending=True,inplace=True)
    # lon = 72.0000000 
    # lat = 26.0000000#大致是一个菱形的中心出处
    crs = ccrs.Orthographic(central_longitude=lon,central_latitude=lat )
    Orthographicpointslon = gpd.GeoSeries(
             [geometry.Point(lon, x) for x in range(-90,90,5) ],crs='EPSG:4326',  # 指定坐标系为WGS 1984
                                )
    Orthographicpoints = gpd.GeoSeries(
             geometry.Point(lon,lat) ,crs='EPSG:4326',  # 指定坐标系为WGS 1984
                                )
    Orthographicpointslat = gpd.GeoSeries(
             [geometry.Point(x, lat) for x in range(-180,180,5) ],crs='EPSG:4326',  # 指定坐标系为WGS 1984
                                )
    gdf = gdf.set_crs('EPSG:4326') 
    gdf= gdf.set_geometry('cell')
    gdf.sort_values('seqnum',ascending=True,inplace=True)
    crs_proj4 = crs.proj4_init##转化为proj4 语句 
    
    # Orthographicpoints_crs = Orthographicpoints.to_crs(crs_proj4)
    # Orthographicpoints_crs.apply(
    #         # lambda x: ax.annotate(text=f'{round(np.rad2deg(self.c_lon),2)} {round(np.rad2deg(self.c_lat),2)}', xy=x.centroid.coords[0], ha='center', fontsize=14,color="r",va= 'bottom') 
    #         lambda x: ax.annotate(text=f'{ round(lon,2) } { round(lat,2)  }', xy=x.centroid.coords[0], ha='center', fontsize=14,color="r",va= 'bottom') 

    #         )
    
    Orthographicpointslat_crs = Orthographicpointslat.to_crs(crs_proj4)
    Orthographicpointslon_crs = Orthographicpointslon.to_crs(crs_proj4)
    
    # Orthographicpointslat_crs.plot(ax = ax ,color = 'r', markersize=20)
    # Orthographicpointslon_crs.plot(ax = ax ,color = 'b', markersize=5)
    # Orthographicpoints_crs.plot(ax = ax ,color = 'r', markersize=20)
    df_ae = gdf.to_crs(crs_proj4)#给geometry换坐标系  
    df_ae.boundary.plot(ax=ax,color='k',zorder=5,alpha=0.7,linewidth =0.8 )
    return ax 
def generate_text( ax ,gdf):
 
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.seqnum):

      text = r"r"+  f"={round(x.corr(y), 2)}"
      # plt.gca().set_color(p1.get_color())
      plt.text(x_lim[1]*0.65 ,x_lim[1]*0.65 , text, ha='center', va='center',fontsize =30,color=color,fontweight=fontweight)
# draw_dggrid_geometry 精简版 只话单独指定的图像 先实验一下不通过经纬度滑块而是通过各自的格网进行滑块会呈现什么样的效果
def draw_dggrid_withsample(res=None,dggtype=None,cnnresultname=None , geostatisname=None):
    
    # 获取深度学习精度
    name = cnnresultname 
    df =pd.read_csv(name)
    df.sort_values('seqnum',ascending=True,inplace=True)
    
    names =[name for name in df.columns if name   in  ['accs','precisions','recalls','f1s',]]
    df_norm =pd.DataFrame()
    for name in names:
        df_norm[f'{name}_norm'] =   df[name] 
    df_norm['seqnum'] = df['seqnum']  
    
    
    #  创建绘制的格网
    gdf=creatcellandcode(dggtype,res)
    gdf = gdf.set_crs('EPSG:4326') 
    gdf= gdf.set_geometry('cell')
    gdf.sort_values('seqnum',ascending=True,inplace=True)
    lon = 72.0000000 
    lat = 26.0000000#大致是一个菱形的中心出处
    crs = ccrs.Orthographic(central_longitude=lon,central_latitude=lat )
    crs1 =ccrs.Robinson(0.0)
    crs_proj4 = crs.proj4_init##转化为proj4 语句 
    # gdf.name =gdf.name.astype(float)
    df_ae = gdf.to_crs(crs_proj4)#给geometry换坐标系 

    # 读取格网的几何属性信息
    df_geo =pd.read_csv(geostatisname)
    values = df_geo['area']
    for name in df_norm.columns:
        if 'seqnum' not in name:
            df_ae[name] =df_norm[name]
    for name in df_norm.columns:
        ax = plt.subplot( projection = crs )
        df_ae.plot(ax=ax,column=name,  legend=True, cmap='OrRd' )
        # draw_dggrid(ax, gdf=gdf)
      
        pointcoord= gdf['point']  ##得到的是-180~180不知道是否会有影响
        pointcoord=np.stack([pointcoord.x,pointcoord.y],axis=1) #n,2 lon,lat
        loc_lon =df_ae.point.x
        loc_lat = df_ae.point.y
        for value,x , y in zip(values,loc_lon,loc_lat):
            plt.text(x ,y, f'{value}', ha='center', va='center',fontsize =2)
        ax.set_title(f'{dggtype}_{res}_{name}')
        os.makedirs('./dggstatispic',exist_ok=True)
        plt.savefig(f'./dggstatispic/{dggtype}_{res}_{name}.png')
        exit()
    return ax 
# draw_same_geometry 的辅助函数 指定colorbar的位置
def add_right_cax(ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距.
    width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)
    return cax
# 把所有格网相同的属性绘制在一起 可以用来绘制精度图
def draw_same_geometry(geopath,res,cellcodepath='./' ):
    df_aes ={}
    names = None
    # proj = ['FULLER' ]
    # topo = ['4D' ]
    for tpo , prj in product(topo,proj ): ##先拓扑后投影 
        dggtype= f'{prj}{tpo}'
        # 加载gdf
        name =f'{dggtype}_{res}_all.csv'
        absname = os.path.join(geopath,name)
        print(absname)
        df =pd.read_csv(absname)
        df.sort_values('seqnum',ascending=True,inplace=True)
        if names is None :
            names =[name for name in df.columns if 'seqnum' not in name]
        df_norm =pd.DataFrame()
        for name in names:
            if 'zsc' in name :
                df_norm[f'{name}_norm'] = df[name]  
            elif 'area' in name:
                df_norm[f'{name}_norm'] =   (df[name]/df[name].mean()).round(3)
            else:
                df_norm[f'{name}_norm'] =   df[name]/df[name].mean()
        df_norm['seqnum'] = df['seqnum']   
            
            # df_norm = pd.DataFrame(scaler.fit_transform(df),columns=df.columns) 
        gdf=creatcellandcode(dggtype,res)
        gdf = gdf.set_crs('EPSG:4326') 
        gdf= gdf.set_geometry('cell')
        gdf.sort_values('seqnum',ascending=True,inplace=True)
        lon = 72.0000000 
        lat = 26.0000000#大致是一个菱形的中心出处
        crs = ccrs.Orthographic(central_longitude=lon,central_latitude=lat )

        for name in df_norm.columns:
            if 'seqnum' not in name:
                gdf[name] =df_norm[name]
                # gdf[f'{name}_norm'] =df_norm[name]
        ax = plt.subplot( projection = crs )
        crs_proj4 = ax.projection.proj4_init##转化为proj4 语句 
        # gdf.name =gdf.name.astype(float)
        df_ae = gdf.to_crs(crs_proj4)#给geometry换坐标系 
        df_aes[f'{dggtype}_{res}'] =df_ae
    for name in names:
        ncols =2
        nrows =3 
        fig,axs = plt.subplots( ncols=ncols,nrows=nrows,subplot_kw={'projection' :crs},figsize=[8*nrows , 8* nrows ]  )
        # fig.subplots_adjust(right=0.9)
        # position = fig.add_axes([0.92, 0.12, 0.015, .78 ])#位置[左,下,右,上]
        # print(axs)
        cmap ='Greys'
        # cmap ='viridis'
        fig.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95,wspace=0.1,hspace=0.1)
        # fig.tight_layout() 
        # max_final = float('-inf') # 无限大 比所有数大
        # min_final = float('inf') #无限小 比所有数小
        for i,(key ,value )in enumerate(df_aes.items()) :
            ax=axs.flatten() [i]
            value:gpd.GeoDataFrame
            cax = add_right_cax(ax, pad=0.02, width=0.02)
            # cax,_ = cbar.make_axes(ax)
            mean = value[f'{name}_norm'].mean()
            std = value[f'{name}_norm'].std()
            # norm = Normalize(vmin=mean-2*std, vmax=mean+2*std)
            ax1 = value.plot(ax=ax,column=f'{name}_norm', cmap=cmap,legend=True,cax=cax,vmin=mean-2.5*std, vmax=mean+2.5*std ) 
            value.boundary.plot(ax=ax,color='k',zorder=5,alpha=1.0,linewidth =0.2 )
            # min_v = value[name].min()
            # max_v = value[name].max()
            # max_final =max(max_v ,max_final)
            # min_final =min(min_v ,min_final)
            ax1.set_title(f'{key}_{name}', fontsize=28)
            # 设置colorbar的属性
            cax.tick_params(labelsize=30)
            ticks = [mean-2*std,mean-std,mean,mean+std,mean+2*std]
            cax.set_yticks([round(value,2) for value in ticks])
            # ax.axis('on')
            # break
            # handles, labels = ax.get_legend_handles_labels()
            # fig.legend(handles, labels, loc='upper center')
        # fig.subplots_adjust(right=0.9)
        # position = fig.add_axes([0.92, 0.12, 0.015, .78 ])#位置[左,下,右,上]
        # print(min_final,max_final)
        # norm = Normalize(vmin=min_final, vmax=max_final)
        # n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
        # n_cmap.set_array([])
        # fig.colorbar(n_cmap, cax=position)

        # fig.tight_layout() 
        fig.savefig(f'{name}.png' )
    # 

# 逐个属性画几何属性图
def draw_dggrid_geometry(dggtype,res,geopath,name,outpath,lon,lat):
    gdf=creatcellandcode(dggtype,res)
    # name =f'{dggtype}_{res}_all.csv'
    absname = os.path.join(geopath,name)
    print(absname)
    df =pd.read_csv(absname)
    drop_data =   df.drop(['seqnum' ,'recalls','precisions','f1s'], axis=1 )
    # scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
    # df_MinMax= pd.DataFrame(scaler.fit_transform(df ), columns=df.columns)
    # scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
    # x 归一化
    # data_x= pd.DataFrame(scaler.fit_transform(data_x ), columns=data_x.columns)

    drop_data =  drop_data.loc[:,~drop_data.columns.str.contains('_cv')]
    drop_data =  drop_data.loc[:,~drop_data.columns.str.contains('_area')]
    drop_data =  drop_data.loc[:,~drop_data.columns.str.contains('per')]
    drop_data =  drop_data.loc[:,~drop_data.columns.str.contains('frac')]
    drop_data =  drop_data.loc[:,~drop_data.columns.str.contains('_std_std')]
    drop_data =  drop_data.loc[:,~drop_data.columns.str.contains('_std_mean')]
    drop_data =  drop_data.loc[:,~drop_data.columns.str.contains('edge_dis')]
    # df_norm = pd.DataFrame(scaler.fit_transform(df),columns=df.columns) 
    # print(drop_data.columns)
    # exit()
    gdf = gdf.set_crs('EPSG:4326') 
    gdf= gdf.set_geometry('cell')
    # gdfout = gpd.drop(gdf,columns=['point'],axis=1)
    gdf.sort_values('seqnum',ascending=True,inplace=True)
    crs = ccrs.Orthographic(lon,lat )
    # crs = ccrs.Gnomonic(central_latitude=lat, central_longitude=lon)s
    # crs = ccrs.LambertAzimuthalEqualArea(central_latitude=lat, central_longitude=lon)
    # crs = ccrs.NorthPolarStereo( central_longitude=lon)
    # crs = ccrs.AzimuthalEquidistant( central_latitude=lat, central_longitude=lon)

    
    crslatlon = ccrs.PlateCarree()
    for name in drop_data.columns:
        # if name in ['hasdata','area_mean']  :
        # if 'area'   in name:
        # if 'seqnum'  not  in name:
            gdf[name] =df[name]
            ax = plt.subplot(  )
            crs_proj4 = crs.proj4_init##转化为proj4 语句 
            # gdf.name =gdf.name.astype(float)
            df_ae = gdf.to_crs(crs_proj4)#给geometry换坐标系 
            gse = gpd.GeoDataFrame(df[name],geometry=gdf['point'],crs=crslatlon)
            # df_ae[f'{name}_bin'], cut_bin = pd.qcut(df_ae[name], q = 5, retbins = True,duplicates='drop')
            # max = df_ae[name].max()
            # min = df_ae[name].min()
            # split =6
            # splitvalue = (max-min)/split
            # bins =[min+splitvalue*i for i in range(split+1)]
            # df_ae[f'{name}_bin'] = pd.cut(x = df_ae[name], bins = bins   )
            # print(df_ae[f'{name}_bin'].value_counts())
            # df_ae.plot(ax=ax,column=f'{name}_bin',  legend=True,   cmap='OrRd' )
            ax1 =df_ae.plot(ax=ax,column=name,  legend=True,   cmap='Blues')
            # geokdeplot(gse,  ax=ax1 ,lon=lon,lat=lat)

            # ax.contourf(df_ae.point.x, df_ae.point.y, df_ae[name], transform=crs, cmap='OrRd', alpha=0.5)
            draw_dggrid(dggtype,0,lon,lat,ax)
            print(f'{lon}_{lat}_{name}')
            ax.set_title(f'{name}')
            # values = np.arange(-6e6,7e6,2e6)
            # xticks = [ crslatlon.transform_point(value,0,src_crs=crs)[0] for value in values]
            # xticks = [ crslatlon.transform_point(0,0,src_crs=crs)[0] for value in values]
            ax.axis('off')
            # plt.xticks()
            os.makedirs(outpath,exist_ok=True)
            plt.tight_layout()
            plt.savefig( os.path.join(outpath,f'{name}_{lon}_{lat}.png'),  transparent=True )
            # exit()
    return ax 

#  借用main函数里的处理数据的方式 这个也是输入随机森林的结果 逐个属性画几何属性图
def draw_dggrid_geometry_1(dggtype,res,geopath,name,outpath,lon,lat):
    gdf=creatcellandcode(dggtype,res)
    # name =f'{dggtype}_{res}_all.csv'
    absname = os.path.join(geopath,name)
    print(absname)
    # 借用main函数里的处理数据的方式 这个也是输入随机森林的结果
    df =dataprocessdata_rdp_nonorm(absname )
    gdf = gdf.set_crs('EPSG:4326') 
    gdf= gdf.set_geometry('cell')
    gdf.sort_values('seqnum',ascending=True,inplace=True)
    gdf= gdf.loc[gdf['seqnum'].isin(list(range(1,1025)))]
    crs = ccrs.Orthographic(lon,lat )
    for name in df.columns:
        # if name in ['hasdata','area_mean']  :
        # if 'area'   in name:
        # if 'seqnum'  not  in name:
            gdf[name] =df[name]
            fig,ax = plt.subplots() 
            crs_proj4 = crs.proj4_init##转化为proj4 语句 
            # gdf.name =gdf.name.astype(float)
            df_ae = gdf.to_crs(crs_proj4)#给geometry换坐标系 
            # df_ae[f'{name}_bin'], cut_bin = pd.qcut(df_ae[name], q = 5, retbins = True,duplicates='drop')
            
            # 分bin绘制 
            # max = df_ae[name].max()
            # min = df_ae[name].min()
            # split =6
            # splitvalue = (max-min)/split
            # bins =[min+splitvalue*i for i in range(split+1)]
            # df_ae[f'{name}_bin'] = pd.cut(x = df_ae[name], bins = bins   )
            # print(df_ae[f'{name}_bin'].value_counts())
            # df_ae.plot(ax=ax,column=f'{name}_bin',  legend=True,   cmap='OrRd' )

            df_ae.plot(ax=ax,column=name,  legend=True,   cmap='gray' )
            
            # ax.contourf(df_ae.point.x, df_ae.point.y, df_ae[name], transform=crs, cmap='OrRd', alpha=0.5)
            draw_dggrid(dggtype,res,lon,lat,ax)
            print(f'{lon}_{lat}_{name}')
            ax.set_title(f'{dggtype}_{res}_{name}')
            os.makedirs(outpath,exist_ok=True)
            # ax.set_xlim(-4e6, 4e6)
            # ax.set_ylim(-5.5e6, 5.5e6)
            # ax.axis('off')
            # plt.subplots_adjust(left=0.1, right=0.9)
            plt.savefig( os.path.join(outpath,f'{dggtype}_{res}_{name}_{lon}_{lat}.png') )
 
            exit()
    return ax 

def main_draw_dggrid_withsample():
    cnnresultname =f'/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_l5/dmdoutbygrid/FULLER4D_5_k1_cnnresults.csv'
    geostatisname =f'/home/dls/data/openmmlab/DGGRID/src/apps/caldgg/c/FULLER4D_5_all.csv'
    draw_dggrid_withsample(cnnresultname=cnnresultname,geostatisname=geostatisname, res=5,dggtype='FULLER4D' )
def main_draw_dggrid_geometry():
    geopath='/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_rdp_l5/dmdoutbygrid'
    name   ='FULLER4D_5_cnngeo.csv'
    outpath ='./rFULLER4D_5_cnngeo'
        # /home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_rdp_l5/dmdoutbygrid/FULLER4D_5_cnngeo.csv
    # geopath = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out'
    # name = 'FULLER4D_5_k2_cnnresults.csv'
    # outpath ='./rFULLER4D_5_k2_cnnresults'
    
    # # geopath = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_l5/dmdoutbygrid'
    # # name ='FULLER4D_5_geostatis.csv' 

    lon = 0
    lat = 90
    # draw_dggrid_geometry_1( geopath=geopath,name=name,outpath=outpath, res=5,dggtype='FULLER4D',lon=lon,lat=lat )
    draw_dggrid_geometry( geopath=geopath,name=name,outpath=outpath, res=5,dggtype='FULLER4D',lon=lon,lat=lat )
import geoplot as gplt
import geoplot.crs as gcrs
import seaborn as sns
def geokdeplot(gdf, ax,lon,lat):
    crs =  gcrs.Orthographic(central_longitude=lon, central_latitude=lat)
    # ax = gplt.kdeplot(df=gdf,kwargs={'ax':ax,'projection':crs}  )
    ax = gplt.kdeplot(df=gdf, ax=ax,projection=crs )
    return ax
def drawcorrel():
    
    dggtype = 'FULLER4D'
    res = 5
    processname = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_rdp_l5/dmdoutbygrid/FULLER4D_5_cnngeo.csv'
    data  =calrel2(processname=processname)

    # 获取所有列名
    cols = data.columns.tolist()

    # 创建子图
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 20))
    sns.color_palette("Paired")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="white", rc=custom_params)
    # 循环绘制子图
    for i, ax in enumerate(axes.flatten()):
        # 如果是最后一行，则只绘制一列
        axout = sns.regplot(data=data, x=cols[i], y='accs',  ax=ax,scatter_kws={"s": 20, 'alpha':.5 })
        x_range = axout.get_xlim()[1] - axout.get_xlim()[0]
        y_range = axout.get_ylim()[1] - axout.get_ylim()[0]
        value = round(data[cols[i]].corr(data['accs']), 2)
        text = r"r"+  f"={value}"
        # plt.gca().set_color(p1.get_color())
        # plt.text(x_lim[1]*0.65 ,x_lim[1]*0.65 , text, ha='center', va='center',fontsize =30  )
        axout.text( axout.get_xlim()[0] +x_range*0.85 ,axout.get_ylim()[0] +y_range*0.9 ,   text, ha='center', va='center',fontsize =18,fontweight='heavy'  )
        ax.tick_params(labelsize=18) #刻度字体大小13
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(16)
        # mean =data[cols[i]].mean()
        # std = data[cols[i]].std()
        min = data[cols[i]].min()
        max = data[cols[i]].max()
        values = np.linspace(min,max,4)
        if i %3 ==0:
            ax.set_ylabel('accs',fontsize=18)
        else:
            ax.set_ylabel('' )
        # 设置colorbar的属性
        # ticks = [mean-2*std,mean-std,mean,mean+std,mean+2*std]
        ax.set_title(' ',pad=2)
        ax.set_xticks([round(value,4) if value<1.0 else round(value,2) for value in values ])
    # 调整子图间距
    fig.tight_layout(rect=(0.01, 0, 1, 1) )
    path ='/home/dls/data/openmmlab/test_RandomForestRegressor/calrel/drawcorrel1.png'
    # 显示图形
    plt.savefig(path)
def drawcorrelxx():
    dggtype = 'FULLER4D'
    res = 5
    processname = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_rdp_l5/dmdoutbygrid/FULLER4D_5_cnngeo.csv'
    data  =calrel2(processname=processname)
    # 获取所有列名
    # sns.set(font_scale=1.1)
    # sns.color_palette("Paired")
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_theme(style="white", rc=custom_params)
    cols = data.columns.tolist()
    def generate_text(x,y, **kwargs):
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="white", rc=custom_params)
        # print(kwargs)
        scaax = plt.gca()
        # print(scaax .yaxis.get_label())
        sns.regplot(x=x,y=y, scatter_kws={"s": 20, 'alpha':.5 })
        # x_lim = (-0.05,1.1)
        # scaax.set_xlim(x_lim)
        # scaax.set_ylim(y_lim)
        x_range = scaax.get_xlim()[1] - scaax.get_xlim()[0]
        y_range = scaax.get_ylim()[1] - scaax.get_ylim()[0]
        value = round(x.corr(y), 2)

        text = r"$\rho$"+  f"={round(x.corr(y), 2)}"
        # plt.gca().set_color(p1.get_color())
        # plt.text(x_lim[1]*0.65 ,x_lim[1]*0.65 , text, ha='center', va='center',fontsize =30  )
        plt.text( scaax.get_xlim()[0] +x_range*0.85 ,scaax.get_ylim()[0] +y_range*0.9 , text, ha='center', va='center',fontsize =15  )
        plt.tight_layout()
    # 创建子图
    # 循环绘制子图
    for num,col in enumerate(cols[:-1]) :
        plt.figure(num)
        # color= np.array([212, 233, 217,255.0])/255.0
        g = sns.JointGrid(data=data, x=col,y=cols[-1])
        # g.plot_joint(sns.regplot, scatter_kws={"s": 20, 'alpha':.5 },ci=98)
        g.plot_joint(generate_text, scatter_kws={"s": 20, 'alpha':.5 } )

        g.plot_marginals(sns.histplot, kde=True)
        # generate_text(data[col],data[cols[-1]])
        # sns.jointplot(data=data,x=col,y=cols[-1],kind='reg',color=color) 
        # plt.tight_layout()
        print(col)
        plt.savefig(f"calrel/{col}.png")


def drawhistplot():
    dggtype = 'FULLER4D'
    res = 5
    processname = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_rdp_l5/dmdoutbygrid/FULLER4D_5_cnngeo.csv'
    data  =calrel2(processname=processname)

    # 获取所有列名
    cols = data.columns.tolist()

    # 创建子图
    # fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8, 8))
    sns.color_palette("Paired")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="white", rc=custom_params)
    # 循环绘制子图
    for num,col in enumerate(cols ) :
            plt.figure(num)
            color= np.array([73, 151, 201,255.0 ])/255.0
            sns.histplot(data=data,x=col, kde=True,bins=15 ,color=color,facecolor=color)
            # generate_text(data[col],data[cols[-1]]) ,hist_kws=dict(facecolor=color, linewidth=2)
            # sns.jointplot(data=data,x=col,y=cols[-1],kind='reg',color=color) 
            # plt.tight_layout()
            print(col)
            plt.ylabel('' )
            plt.xlabel('' )
            plt.title(col)
            plt.savefig(f"calrel/{col}_hist.png",   transparent=True )
    # path ='/home/dls/data/openmmlab/test_RandomForestRegressor/calrel/drawcorrel.png'
    # # 显示图形
    # plt.savefig(path)
def main ():
    # rdp数据处理步骤
    # 先计算cnn结果的均值

    path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_rdp_l5/dmdoutbygrid'
    kolds = ['k0','k1','k2','k3','k4']
    # cnn_meanresult(path =path ,kolds=kolds)
    #  把均值结果和几何属性结果进行合并
    # merge_geo_metric(outpath=path)
    # 计算整个格网的均值 让processdat调用
    dggridpath = '/home/dls/data/openmmlab/DGGRID/src/apps/caldgg/c'
    calmeanoutpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/mean'
    # calmean(geopath=dggridpath,calmeanoutpath=calmeanoutpath)
    #  读取合并的文件进行相关性钱的数据归一化准备
    processdata_outpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/processdata'
    os.makedirs(processdata_outpath,exist_ok=True)
    outdirname =f'FULLER4D_5_process.csv'
    # processdata_rdp(cnngeopath =path,outpath =processdata_outpath,outdirname=outdirname)
    # 计算相关性
    calreloutpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/calrel'
    outname =  '/home/dls/data/openmmlab/test_RandomForestRegressor/calrel/outfilterbyreal.csv'
    calreldirname =f'FULLER4D_5_rel_nocv_maxscaler.csv'
    processdatafile =os.path.join(processdata_outpath,outdirname)
    # calrel(processdatafile,outpath=calreloutpath,dirname=calreldirname)

    processdatafile2 = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_rdp_l5/dmdoutbygrid/FULLER4D_5_cnngeo.csv'
    calrel2(processdatafile2,outname=outname )
    # 计算因子分析 用原始数据进行因子分析
    # filename = '/home/dls/data/openmmlab/DGGRID/src/apps/caldgg/c/FULLER4D_5_all.csv'
    # factor_ana(filename)
    # process_outpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/b60_l5_ratio'
   
    # merge_file_name ='merge_b60_noscaler.csv'
    # res =5
    # # processdata(process_outpath)
    # # mergedata(process_outpath,merge_file_name=merge_file_name)
    # for prj , tpo in product(proj,topo):
    #     print(f'{prj}{tpo}_{res}')
    #     dirname = os.path.join(process_outpath,f"{prj}{tpo}_{res}_process_noscaler.csv")
    #     get_xarray(dirname,f'{prj}{tpo}_{res}')
        # plt.show()
        # time.sleep(2)
        # exit()
    # calrel('/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/large_samples_b60_scaler/FULLER4D_6_process_sacler.csv')
    # calrel('/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/large_samples_b60_noscaler/FULLER4D_6_process_nosacler.csv')
    # ax = plt.subplot( projection=ccrs.Robinson(0.0) ,figsize=(10, 10))
    # crs = ccrs.Orthographic(0.0,0.0 )
    # crs1 =ccrs.Robinson(0.0)
    # res =2 
    
    # 绘制一个层级格网的属性分布
        # geopath = '/home/dls/data/openmmlab/DGGRID/src/apps/caldgg/c' ##存储文件夹 目前只有第六层的
        # for prj , tpo in product(proj,topo):
        #     print(f'{prj}{tpo}_{res}')
        #     dggtype = f'{prj}{tpo}'
        #     res =6 
        #     draw_dggrid_geometry(dggtype,res,geopath)
    # fig, ax = plt.subplots(subplot_kw={'projection':  crs},figsize=(10, 10))
    # cell_file_name = '/home/dls/data/openmmlab/mmclassification/data/sphere/temp/temp_ISEA4D_4_out_87ad92aa-fc83-4a42-be78-665979b04671.json'
    # point ='/home/dls/data/openmmlab/mmclassification/data/sphere/temp/temp_ISEA4D_4_out_87ad92aa-fc83-4a42-be78-665979b04671_point.json'
    # # draw_dggrid(ax,res)
    # geopath = '/home/dls/data/openmmlab/DGGRID/src/apps/caldgg/c' ##存储文件夹 目前只有第六层的
    # draw_same_geometry(geopath,res=5)

if __name__ == '__main__':
    # main ()
    # main_draw_dggrid_geometry()
    # drawcorrel1()
    drawcorrel()
    # drawhistplot()
    # path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out'
    # if res  ==6 :
    #     path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/outl6'
    # # cnn_meanresult(path)
    # merge_geo_metric(path)
    
    # # main()
    # # 计算格网在固定区域的测试值 操作是吧json转化为csv
    
    # if res == 6 :
    #     ori_path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out_0_0'
    # else :
    #     ori_path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out_ori'
    # print(f'res:{res} ,ori_path:{ori_path}')
    # cnn_meanresultori(ori_path)
    
    # # # # 计算每个格网属性的几何平均值
    # # geopath = '/home/dls/data/openmmlab/DGGRID/src/apps/caldgg/c'
    # # calmean(geopath)
    # # # 合并 几何属性 和深度学习精度文件
    # if res == 6:
    #     out_process_path= '/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/b60_l6_ratio'
    #     cnngeopath= '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/outl6'
        
    # else :
    #     out_process_path= '/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/b60_l5_ratio'
    #     cnngeopath= '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out'
    # processdata(out_process_path,cnngeopath)
    
    # # # # 把处理好的数据合并成一个文件 让随机森林调用
    
    # if res == 6:
    #     out_process_path= '/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/b60_l6_ratio'
    # else :
    #     out_process_path= '/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/b60_l5_ratio'
    # mergeprocessdata(out_process_path)
    
    # # # 首先计算相关性 '/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/b60_l5_ratio/merge_process_noscaler.csv',
    # dirname='/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/b60_l6_ratio/merge_process_noscaler_6.csv'
    # print(dirname)
    # outpath ='./calrel/all_6' 
    # calrel(dirname,outpath=outpath)   
    # dirnames = ['/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/b60_l5_ratio/FULLER4D_5_process_noscaler.csv',
    #             '/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/b60_l5_ratio/FULLER4H_5_process_noscaler.csv',
    #             '/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/b60_l5_ratio/FULLER4T_5_process_noscaler.csv'
    #             ]       
    # outpaths =[
    #     './calrel/FULLER4D_5',
    #     './calrel/FULLER4H_5',
    #     './calrel/FULLER4T_5',
    # ]
    # for i,dirname in enumerate( dirnames):
    #     print(dirname)
        # calrel(dirname,outpaths[i])
        # exit()