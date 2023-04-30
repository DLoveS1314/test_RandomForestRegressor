# 除以全格网的算术平均值 然后逐个文件进行归一化

from itertools import product
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from itertools import product
import json
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.colorbar as cbar
import matplotlib as mpl
from maketablefromfile import *
from itertools import product
import time
path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out'
proj = ['FULLER','ISEA']
topo = ['4D','4H','4T']
res = 5
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
def cnn_meanresult(path):
# 处理由mmcls生成的数据 对于交叉验证精度表格要进行取平均后再与几何属性进行合并
    kolds = ['k0','k1','k2']
    for prj , tpo in product(proj,topo):
        dfs = []
        for k in kolds:
            df= pd.read_csv(os.path.join(path,f"{prj}{tpo}_{res}_{k}_cnnresults.csv"))
            round_name = [name for name in df.columns if 'names' not in name]
            # 不需要那么多位数
            # df[round_name] = df[round_name].round(5)
            dfs.append(df)
        outpath = os.path.join(path,f"{prj}{tpo}_{res}_cnnmean.csv")
        df_con = pd.concat(dfs, axis=0, ignore_index=True)
        # meanname = [name for name in df_con.columns if name not in ['names','r_lon','r_lat']]
        df_group = df_con.groupby([ 'names','r_lon','r_lat']) 
        print(f'groupby names {len(df_group.groups.keys())}')
        df_group_mean = df_group.mean()
        df_group_mean.reset_index(inplace=True)
        df_group_mean = df_group_mean.rename(columns={'names': 'name'})
        df_group_mean.to_csv(outpath,index=False) 
        print(f'cnn_meanresult save result to {outpath}')

def merge_geo_metric(outpath):
    # 如果改变了统计量 需要重新合并/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out 里的 calstatis 和 metrics 文件变为all文件 与savepad功能相同
    # path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out'
    # proj = ['FULLER','ISEA']
    # topo = ['4D','4H','4T']
    for prj , tpo in product(proj,topo):
        
        dirname_calstatis = os.path.join(outpath,f"{prj}{tpo}_{res}_geostatis.csv")
        dirname_metircs = os.path.join(outpath,f"{prj}{tpo}_{res}_cnnmean.csv")
        dirname_all = os.path.join(outpath,f"{prj}{tpo}_{res}_cnngeo.csv")
        
        df_cal =pd.read_csv(dirname_calstatis)
        print(df_cal.head())
        df_met =pd.read_csv(dirname_metircs)
        # print(df_met.head())
        
        merged_df = pd.merge(df_cal, df_met, on='name', how='inner')
        # print(merged_df.columns)
        # print(f'save result to {dirname_all}')
        merged_df.to_csv(dirname_all,index=False)
# 处理 由mmcls生成的数据（ 包含几何属性 和 预测精度 ）删除不需要的属性 并进行归一化
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
# 把processdata 处理好的文件进行合并 用于随机森林训练
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

# 转化为xarray 画图 方便的是可以直接进行经纬度的画图 用于画f1score比较好
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
# 目的是删除一些不与line相交的格网 这各函数是辅助函数 返回相交的格网有哪些
def check_intersects(geometry,line):
    return geometry.apply(lambda geom: geom.intersects(line))
# 给一个格网属性或者gdf 画一个格网图
def draw_dggrid(ax,res=None,dggtype=None,gdf=None):
    if gdf is None :
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
    df_ae.boundary.plot(ax=ax,color='r',zorder=5,alpha=0.6,linewidth =0.3 )

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
# 把所有格网相同的属性绘制在一起
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
            gdf[name] =df[name]
            ax = plt.subplot( projection = crs )
            crs_proj4 = ax.projection.proj4_init##转化为proj4 语句 
            # gdf.name =gdf.name.astype(float)
            df_ae = gdf.to_crs(crs_proj4)#给geometry换坐标系 
            df_ae.plot(ax=ax,column=name,  legend=True, cmap ='gray')
            # draw_dggrid(ax, gdf=gdf)
            ax.set_title(f'{dggtype}_{res}_{name}')
            os.makedirs('./nonorm',exist_ok=True)
            plt.savefig(f'./nonorm_geopic/{dggtype}_{res}_{name}.png')
    return ax 
# 计算指标之间的相关性
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
        # geopath = '/home/dls/data/openmmlab/DGGRID/src/apps/caldgg/c' ##存储文件夹 目前只有第六层的
        # for prj , tpo in product(proj,topo):
        #     print(f'{prj}{tpo}_{res}')
        #     dggtype = f'{prj}{tpo}'
        #     res =6 
        #     draw_dggrid_geometry(dggtype,res,geopath)
    # fig, ax = plt.subplots(subplot_kw={'projection':  crs},figsize=(10, 10))
    cell_file_name = '/home/dls/data/openmmlab/mmclassification/data/sphere/temp/temp_ISEA4D_4_out_87ad92aa-fc83-4a42-be78-665979b04671.json'
    point ='/home/dls/data/openmmlab/mmclassification/data/sphere/temp/temp_ISEA4D_4_out_87ad92aa-fc83-4a42-be78-665979b04671_point.json'
    # draw_dggrid(ax,res)
    geopath = '/home/dls/data/openmmlab/DGGRID/src/apps/caldgg/c' ##存储文件夹 目前只有第六层的
    draw_same_geometry(geopath,res=5)
if __name__ == '__main__':
    path = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/out'
    cnn_meanresult(path)
    merge_geo_metric(path)