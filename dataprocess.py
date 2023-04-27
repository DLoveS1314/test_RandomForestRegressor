# 除以全格网的算术平均值 然后逐个文件进行归一化

from itertools import product
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from itertools import product
import json
import geopandas as gpd
from shapely.geometry import Point
path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out'
proj = ['FULLER','ISEA']
topo = ['4D','4H','4T']
res = 6
# path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out'
path = '/home/dls/data/openmmlab/mmclassification/tools/large_samples_b60/out'

json_path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out_0_0'
# 处理json数据为df文件
def gettestscore(json_dirname):
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
def processdata(outpath):
    for prj , tpo in product(proj,topo):
        dirname = os.path.join(path,f"{prj}{tpo}_{res}_all.csv")
 
        print(f'dirname: {dirname}')
        df =pd.read_csv(dirname)
        # 筛选包含 mean 或 std 的列 默认filter列筛选 想行筛选 坐标轴选择 axis=0
        columns_with_mean = df.filter(like='_mean').columns
        columns_with_std = df.filter(like='_std').columns
        #  遍历这些列，计算对应的 mean/std 的值
        # iteritems（） - 遍历列（键，值）对
        # iterrows（） - 遍历行（索引，序列）对
        for column_name, column_data in df[columns_with_mean].items ():#遍历每一列
            stat_name = column_name.split('_')[0]  # 获取统计名称，如 area
            std_column_name = f'{stat_name}_std'  # 构造对应的 std 列名，如 area_std
            if std_column_name in columns_with_std:#如果有对应std名
                std_column_data = df[std_column_name]  # 获取对应的 std 列数据
                df[f'{stat_name}_cv'] = std_column_data / column_data     # 计算 mean/std 的值
        ##删除多余的变量
        # df_drop= df_sorted.drop(['name', 'r_lon', 'r_lat','recalls','accs','f1s'], axis=1 )
        df_drop= df.drop(['name', 'r_lon', 'r_lat' ], axis=1 )
        drop_data = df[['r_lon', 'r_lat' ]]  
        # df_drop= df
        json_dirname = os.path.join(json_path,f"{prj}{tpo}_l{res}_00.json")
        json_dirname=json_dirname.lower()
        pre_test = gettestscore(json_dirname)#0 0 位置的预测精度
        csv_dirname = os.path.join(json_path,"out",f"{prj}{tpo}_{res}_00_calstatis.csv") #0 0 位置处的格网几何属性
        metric_test = pd.read_csv(csv_dirname)
        metric_test= metric_test.drop(['name', 'r_lon', 'r_lat' ], axis=1 )
        for key ,val in metric_test.items():##因为涉及负值 所以用变化率来表示 abs((a-b)/a)
            value = float(metric_test[key]) 
            df_drop[f'{key}_norm'] = abs( (value-df_drop[key])/value ) ##几何属性数据标准化一下 存成对应的norm数据 利用的是 0 0 位置处进行标准化
        for key ,val in pre_test.items():
            value = float(pre_test[key]) 
            # print(key ,value)
            df_drop[f'{key}_norm'] = abs( (value-df_drop[key])/value )##预测精度数据标准化一下 存成对应的norm数据 利用的是 0 0 位置处进行标准化
        csv_mean = os.path.join('/home/dls/data/openmmlab/test_RandomForestRegressor/mean',f"{prj}{tpo}_{res}_mean.csv") #0 0 位置处的格网几何属性
        metric_mean = pd.read_csv(csv_mean)
        for key ,val in metric_mean.items():##因为涉及负值 所以用变化率来表示 abs((a-b)/a)
            value = float(metric_mean[key]) 
            df_drop[f'{key}_meannorm'] = abs( (value-df_drop[key])/value ) ##用算术平均值norm存成对应的norm数据 利用的是全球位置的均值进行标准化
            # 对数据进行归一化 本身就是逐列归一化

        df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_var')]##loc 去除方差行 选择多列和多行
        df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_cv')]##loc 去除方差行 选择多列和多行
        df_drop = df_drop.loc[:,~df_drop.columns.str.contains('max')]##loc 去除方差行 选择多列和多行
        df_drop = df_drop.loc[:,~df_drop.columns.str.contains('min')]##loc 去除方差行 选择多列和多行
        df_drop = df_drop.loc[:,~df_drop.columns.str.contains('maxmin')]##loc 去除方差行 选择多列和多行
        df_drop = df_drop.loc[:,~df_drop.columns.str.contains('_ir')]##loc 去除方差行 选择多列和多行
        df_drop = df_drop.loc[:,df_drop.columns.str.contains('norm')]##loc 去除方差行 选择多列和多行       
        scaler = MinMaxScaler()
        df_drop = df_drop[sorted(df_drop.columns)]
        # df_drop=pd.concat([df_drop,drop_data],axis=1)
        # outdirname = os.path.join(outpath,f"{prj}{tpo}_{res}_process_nosacler.csv")
        # os.makedirs(outpath,exist_ok=True)
        # df_drop.to_csv(outdirname,index=False)
        df_drop = pd.DataFrame(scaler.fit_transform(df_drop ), columns=df_drop.columns)
        df_drop=pd.concat([df_drop,drop_data],axis=1)
        os.makedirs(outpath,exist_ok=True)
        outdirname = os.path.join(outpath,f"{prj}{tpo}_{res}_process_sacler.csv")
        print(outdirname)
        df_drop.to_csv(outdirname,index=False)
def mergedata(filepath,merge_file_name):
    # 读取文件夹中所有 CSV 文件
    folder_path = filepath
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    # 创建空的 DataFrame
    merged_df = []
    # 循环遍历所有 CSV 文件
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        print('megre file : ',file_path)
        # 读取 CSV 文件并将数据合并到 DataFrame 中
        df = pd.read_csv(file_path)
        merged_df.append(df)
    merged_df = pd.concat(merged_df  )
    merged_df.to_csv( merge_file_name , index=False)
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

import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
def get_xarray(path,titlename=None):
    crs = ccrs.Orthographic(0.0,0.0 )
    crs1 =ccrs.Robinson(0.0)
    ax = plt.subplot( projection=crs1 )
    # fig, ax = plt.subplots(figsize=(4,8))
    name = os.path.basename(path)
    print('name',path)
    df = pd.read_csv(path)
    df = df.set_index(['r_lat','r_lon'])
    ds = xr.Dataset.from_dataframe(df  )
    p = ds['f1s_norm'].plot( transform=ccrs.PlateCarree(),ax=ax)
    # p = ds['f1s_norm'].plot( ax=ax)
    
    # p.axes.set_global()
    ax.set_title(titlename)
    p.axes.gridlines()
    p.axes.coastlines()
    # p.axes.set_extent([120, 250, -75, 75], crs=ccrs.PlateCarree())
    # draw_dggrid(p.axes,2)
    # draw_dggrid(p.axes,0)

    
    p.figure.savefig(f'{name}_xr.png')
from shapely.geometry import Polygon, LineString, Point

from maketablefromfile import *
def check_intersects(geometry,line):
    return geometry.apply(lambda geom: geom.intersects(line))
def draw_dggrid(ax,res,dggtype=None):
    gdf=creatcellandcode(dggtype,res)
    gdf = gdf.set_crs('EPSG:4326') 
    gdf= gdf.set_geometry('cell')
    # line = gpd.GeoSeries(  [LineString([(180, -89), (180, 89)]),
    #                       LineString([(-180, -89), (-180, 89)])])
    # line = gpd.GeoSeries(  
    #                       LineString([(-180, -89), (-180, 89)]))
    
    # line1 = gpd.GeoSeries(   LineString([(180, -89), (180, 89)]) 
    #                        )
    # line = line.set_crs('EPSG:4326') 
    
    # df_i = check_intersects(gdf.geometry,line)
    # df_i1 = check_intersects(gdf.geometry,line1)
    # print(df_i.to_numpy().shape)
    # boolvalue =(df_i.to_numpy().squeeze()== False)  & (df_i1.to_numpy().squeeze() ==False)
    # print(boolvalue)
    # gdf = gdf[boolvalue]

    # 删除包含180度经度线的多边形列
    # gdf = gdf.drop("contains_180", axis=1)
    
    # This can be converted into a `proj4` string/dict compatible with GeoPandas
    crs_proj4 = ax.projection.proj4_init##转化为proj4 语句 
    # gdf.name =gdf.name.astype(float)
    df_ae = gdf.to_crs(crs_proj4)#给geometry换坐标系 
    df_ae.boundary.plot(ax=ax,color='r',zorder=5,alpha=1.0,linewidth =0.6 )

def draw_dggrid_geometry(dggtype,res,geopath):
    gdf=creatcellandcode(dggtype,res)
    name =f'{dggtype}_{res}_all.csv'
    absname = os.path.join(geopath,name)
    print(absname)
    df =pd.read_csv(absname)
    df.sort_values('seqnum')
    scaler =MinMaxScaler()
    
    df_norm = pd.DataFrame(scaler.fit_transform(df),columns=df.columns) 
    gdf = gdf.set_crs('EPSG:4326') 
    gdf= gdf.set_geometry('cell')
    gdf.sort_values('seqnum',ascending=True,inplace=True)
    crs = ccrs.Orthographic(0.0,0.0 )
    crs1 =ccrs.Robinson(0.0)

    for name in df_norm.columns:
        if 'seqnum' not in name:
            gdf[name] =df_norm[name]
            ax = plt.subplot( projection = crs )
            crs_proj4 = ax.projection.proj4_init##转化为proj4 语句 
            # gdf.name =gdf.name.astype(float)
            df_ae = gdf.to_crs(crs_proj4)#给geometry换坐标系 
            df_ae.plot(ax=ax,column=name,  legend=True, cmap ='gray')
            draw_dggrid(ax,res=1,dggtype=dggtype)
            ax.set_title(f'{dggtype}_{res}_{name}')
            plt.savefig(f'{dggtype}_{res}_{name}.png')
    return ax 
def calrel(path):
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_rows', 1000)
    data = pd.read_csv(path)
    data =data.drop(['r_lat','r_lon'],axis=1)
    data_y=[ 
    'recalls_norm',
    'accs_norm',
    'f1s_norm',
    'precisions_norm']
    x_name = [name for name in data.columns if name not in data_y]
    data_x =  data[x_name]
    data_y =  data[data_y]
    # need_name = ['area','per','zsc','disminmax','disavg','angleminmax','angleavg','csdminmax','csdavg','precisions' ]
    # data =data[need_name]
    # 将数据集分成训练集和测试集 最后一列是因变量 剩余的列是自变量
    scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
    scaler.fit(data_x)
    scaler.fit(data_y)
    df_normalized= pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns)
    # df_normalized.head()
    name = path.split('.')[0]
    df_normalized.to_csv(f'{name}_norm.csv',index=False)
    corr = data_x.corrwith(data_y['f1s_norm'] )
    print(
            f'Correlation between data_x and f1s_norm :\n{corr }\n'
        )
    # 发现norm完后的结果普遍变好 利用spherephd处理的结果没有普通的好

    corr1 = df_normalized.corrwith(data_y['f1s_norm'] )
    print(
        f'Correlation between df_normalized and f1s_norm :\n{corr1 }\n'
    )
def factor_ana():
    pass
import time
def main ():
    process_outpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/large_samples_b60_scaler'
   
    merge_file_name ='merge_b60_noscaler.csv'
    res =6
    # processdata(process_outpath)
    # mergedata(process_outpath,merge_file_name=merge_file_name)
    # for prj , tpo in product(proj,topo):
    #     print(f'{prj}{tpo}_{res}')
    #     dirname = os.path.join(process_outpath,f"{prj}{tpo}_{res}_process_sacler.csv")
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
    geopath = '/home/dls/data/openmmlab/DGGRID/src/apps/caldgg/c' ##存储文件夹 目前只有第六层的
    for prj , tpo in product(proj,topo):
        print(f'{prj}{tpo}_{res}')
        dggtype = f'{prj}{tpo}'
        res =6 
        draw_dggrid_geometry(dggtype,res,geopath)
    # fig, ax = plt.subplots(subplot_kw={'projection':  crs},figsize=(10, 10))
    cell_file_name = '/home/dls/data/openmmlab/mmclassification/data/sphere/temp/temp_ISEA4D_4_out_87ad92aa-fc83-4a42-be78-665979b04671.json'
    point ='/home/dls/data/openmmlab/mmclassification/data/sphere/temp/temp_ISEA4D_4_out_87ad92aa-fc83-4a42-be78-665979b04671_point.json'
    # draw_dggrid(ax,res)
    
if __name__ == '__main__':
    main()