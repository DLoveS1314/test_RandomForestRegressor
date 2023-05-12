from sklearn.ensemble import RandomForestRegressor  ,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from save_model import *
import time
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier 
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.model_selection import ParameterGrid
# from sklearn.metrics import 
from sklearn.metrics import ConfusionMatrixDisplay  
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

randomseed = 42
import xgboost as xgb

# rdp 没有固定区域的cnn结果

def plotbins(processname):
    pass
    dirname = processname
    df =pd.read_csv(dirname)
    df_split= df.loc[df['seqnum'].isin(list(range(1,1025)))]
    nbins =20
    kbins = KBinsDiscretizer(n_bins=nbins, strategy='uniform',encode='ordinal')
    # kbins = KBinsDiscretizer(n_bins=10, strategy='uniform',encode='onehot')
    y_binned = kbins.fit_transform(df_split['accs'].to_numpy() .reshape(-1, 1) ) .squeeze()
    sns.histplot(df_split['accs'], kde=True,bins=nbins )
    plt.savefig( f'y_binned{nbins}.png')
def dataprocessdata_rdp( processname ) :
        dirname = processname
        df =pd.read_csv(dirname)
        df_split= df.loc[df['seqnum'].isin(list(range(1,1025)))]
        drop_data = df['seqnum'] 

        # scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
        # df_MinMax= pd.DataFrame(scaler.fit_transform(df ), columns=df.columns)
        scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
        df_MinMax= pd.DataFrame(scaler.fit_transform(df_split ), columns=df_split.columns)
        # df_drop= df_sorted.drop(['name', 'r_lon', 'r_lat','recalls','accs','f1s'], axis=1 )
        df_drop_MinMax= df_MinMax.drop(['seqnum' ,'recalls','precisions','f1s'], axis=1 )
         
        # 删除不需要的列 相关性小于一定值的列进行排除
        data_y_name =[ 'accs']
        # df_normalized= pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        # scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
        # df_MinMax= pd.DataFrame(scaler.fit_transform(df_drop_split), columns=df_drop_split.columns)
 
        # # x_name = [name for name in data.columns if (name not in data_y_name) and ('_cv' not in name) and ('_meanratio' in name) ]
        x_name = [name for name in df_drop_MinMax.columns if (name not in data_y_name)  ]

        data_x =  df_drop_MinMax[x_name]
        data_y =  df_drop_MinMax[data_y_name]
        # 删除列 
        data_x =  data_x.loc[:,~data_x.columns.str.contains('_cv')]
        data_x =  data_x.loc[:,~data_x.columns.str.contains('_area')]
        data_x =  data_x.loc[:,~data_x.columns.str.contains('per')]
        data_x =  data_x.loc[:,~data_x.columns.str.contains('frac')]
        data_x =  data_x.loc[:,~data_x.columns.str.contains('_std_std')]
        data_x =  data_x.loc[:,~data_x.columns.str.contains('_std_mean')]
        data_x =  data_x.loc[:,~data_x.columns.str.contains('edge_dis')]
 
        # print('data_x',data_x.columns)
        # print('data_y',data_y.columns)

        corr0 = data_x.corrwith(data_y['accs']  )
        print(corr0[abs(corr0)>0.4])
        columnsx = list(corr0.index[abs(corr0)>0.4])
        columns =columnsx+['accs']
        df_all = pd.concat([ drop_data,data_x,data_y ],axis=1)
        df_all = df_all[sorted(df_all.columns)]
        df_all_split= df_all.loc[df_all['seqnum'].isin(list(range(1,1025)))]
        # 删除seqnum
        df_all_split= df_all_split.drop(['seqnum'  ], axis=1 )
        df_all_split.to_csv(f'dataprocessdata_rdp_all_f1s.csv', index=False)
        
        return  df_all_split[columns]
   
        # coors[f'coors'] =corr0
        # coors_filter = coors[coors['coors']>0.4]
        # print(coors_filter)
        # maxscaler = MinMaxScaler()
        # meanscaler = StandardScaler()
        # df_drop_maxsacler= pd.DataFrame(maxscaler.fit_transform(df_drop ), columns=[f'{column}_maxscaler'for column in df_drop.columns ])
        # df_drop_meansacler= pd.DataFrame(meanscaler.fit_transform(df_drop ), columns=[f'{column}_meanscaler'for column in df_drop.columns ])
        # # df_all = pd.concat([drop_data,df_drop,df_meannorm,df_drop_maxsacler,df_drop_meansacler],axis=1)
        # df_all = pd.concat([drop_data,df_drop, df_drop_maxsacler,df_drop_meansacler],axis=1)
        # df_all = df_all[sorted(df_all.columns)]
        # print(f'dirname: {dirname} outdirname :{outdirname}')

# 因变量不作归一化
def dataprocessdata_rdp_normx( processname ) :
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
        scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
        # x 归一化
        data_x= pd.DataFrame(scaler.fit_transform(data_x ), columns=data_x.columns)

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
        df_all = pd.concat([  data_x,data_y ],axis=1)
        df_all = df_all[sorted(df_all.columns)]
        df_all.to_csv(f'dataprocessdata_rdp_all_f1s.csv', index=False)
        
        return  df_all[columns]
   
        # coors[f'coors'] =corr0
        # coors_filter = coors[coors['coors']>0.4]
        # print(coors_filter)
        # maxscaler = MinMaxScaler()
        # meanscaler = StandardScaler()
        # df_drop_maxsacler= pd.DataFrame(maxscaler.fit_transform(df_drop ), columns=[f'{column}_maxscaler'for column in df_drop.columns ])
        # df_drop_meansacler= pd.DataFrame(meanscaler.fit_transform(df_drop ), columns=[f'{column}_meanscaler'for column in df_drop.columns ])
        # # df_all = pd.concat([drop_data,df_drop,df_meannorm,df_drop_maxsacler,df_drop_meansacler],axis=1)
        # df_all = pd.concat([drop_data,df_drop, df_drop_maxsacler,df_drop_meansacler],axis=1)
        # df_all = df_all[sorted(df_all.columns)]
        # print(f'dirname: {dirname} outdirname :{outdirname}')

# 自变量和因变量都不作归一化
def dataprocessdata_rdp_nonorm( processname ) :
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
        df_all = pd.concat([  data_x,data_y ],axis=1)
        # df_all = df_all[sorted(df_all.columns)]
        df_all.to_csv(f'dataprocessdata_rdp_all_f1s.csv', index=False)
        
        return  df_all[columns]
   
        # coors[f'coors'] =corr0
        # coors_filter = coors[coors['coors']>0.4]
        # print(coors_filter)
        # maxscaler = MinMaxScaler()
        # meanscaler = StandardScaler()
        # df_drop_maxsacler= pd.DataFrame(maxscaler.fit_transform(df_drop ), columns=[f'{column}_maxscaler'for column in df_drop.columns ])
        # df_drop_meansacler= pd.DataFrame(meanscaler.fit_transform(df_drop ), columns=[f'{column}_meanscaler'for column in df_drop.columns ])
        # # df_all = pd.concat([drop_data,df_drop,df_meannorm,df_drop_maxsacler,df_drop_meansacler],axis=1)
        # df_all = pd.concat([drop_data,df_drop, df_drop_maxsacler,df_drop_meansacler],axis=1)
        # df_all = df_all[sorted(df_all.columns)]
        # print(f'dirname: {dirname} outdirname :{outdirname}')


def dataprocess(path,metircs,usearea=False,usenorm=False ):
 
    # NOTE usenorm 表示是否使用归一化 而不是表示是都使用带norm的自变量！！！
    data = pd.read_csv(path)
    data_y_name=['recalls',
    'recalls_norm',
    'accs',
    'accs_norm',
    'f1s',
    'f1s_norm',
    'precisions',
    'precisions_norm']
    x_name = [name for name in data.columns if name not in data_y_name]
    data_x =  data[x_name]
    data_y =  data[metircs]
    if usearea:
        namefilter =[name for name in data_x.columns if (('norm'in name) and ('_ir' not in name)  ) ]
    else :
        namefilter =[name for name in data_x.columns if (('norm'in name) and ('_ir' not in name) and ('_area' not in name)) ]
    data_x = data_x[namefilter]##loc 去除方差行 选择多列和多行
    # exit()
    # need_name = ['area','per','zsc','disminmax','disavg','angleminmax','angleavg','csdminmax','csdavg','precisions' ]
    # data =data[need_name]
    # 将数据集分成训练集和测试集 最后一列是因变量 剩余的列是自变量
    scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
    df_normalized = pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns)
    # X_train, X_test, y_train, y_test = train_test_split(df_normalized, data_y, test_size=0.2, random_state=19960229)
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=randomseed)
    if usenorm :
        X_train, X_test, y_train, y_test = train_test_split(df_normalized, data_y, test_size=0.2, random_state=randomseed)
    # X_train.to_csv(os.path.join(savedir,'X_train.csv') ,index=False)
    # X_test.to_csv(os.path.join(savedir,'X_test.csv') ,index=False)
    # y_train.to_csv(os.path.join(savedir,'y_train.csv') ,index=False)
    # y_test.to_csv(os.path.join(savedir,'y_test.csv') ,index=False)
    return X_train, X_test, y_train, y_test
# 加载数据集
def run( model,param_grid,data  ):
    X_train, X_test, y_train, y_test = data
    # 网格搜索
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=12, verbose=1,return_train_score=True)
    grid_search.fit(X_train, y_train)
# XGBoost  结合 sklearn使用 GridSearchCV
    # 输出最佳参数组合
    # name = 'rf_grid_sea.pkl'

    # 用最佳参数组合训练模型
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train, y_train)
    # 用最佳参数组合对测试集进行预测
    y_pred = best_rf.predict(X_test)

    # 计算R2和RMSE
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # csv_name = basename+f'{timestamp}_{round(r2,3)}.csv'

    return grid_search ,r2 ,rmse,best_rf


    # # 保存权重
    # import joblib
    # joblib.dump(gbt, 'rf_regressor.pkl')

def runclass( model,param_grid,data  ):
    X_train, X_test, y_train, y_test = data
    # 计算类别权重
    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

    # 将类别权重传递给模型
 
    # 网格搜索
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=12, verbose=1,return_train_score=True)
    grid_search.fit(X_train, y_train)
# XGBoost  结合 sklearn使用 GridSearchCV
    # 输出最佳参数组合
    # name = 'rf_grid_sea.pkl'

    # 用最佳参数组合训练模型
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train, y_train,class_weight)
    # 用最佳参数组合对测试集进行预测
    y_pred = best_rf.predict(X_test)

    # 计算R2和RMSE
    acc = best_rf.score(y_test, y_pred)
  

    # csv_name = basename+f'{timestamp}_{round(r2,3)}.csv'

    return grid_search ,acc , best_rf


    # # 保存权重
    # import joblib
    # joblib.dump(gbt, 'rf_regressor.pkl')

def savedata(data,datapath ):

    X_train, X_test, y_train, y_test =data 
    os.makedirs(datapath,exist_ok=True)
    X_train.to_csv(os.path.join(datapath,'X_train.csv') ,index=False)
    X_test.to_csv(os.path.join(datapath,'X_test.csv') ,index=False)
    y_train.to_csv(os.path.join(datapath,'y_train.csv') ,index=False)
    y_test.to_csv(os.path.join(datapath,'y_test.csv') ,index=False)



def mainxboost(usearea,path,tag='largeb60', metircs='f1s_norm',  usenorm = True):
    # 设置 XGBoost 模型的参数空间
    # define XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=randomseed
    )
    param_grid = {
        'max_depth': [ 7,10,None ],
        'learning_rate': [0.05,0.01, 0.03 ],
        # 'booster':['gbtree','gblinear'],
        'n_estimators': [ 150,300,500 ],
        'subsample': [ 0.8, 0.9],
        'colsample_bytree': [ 0.8, 0.9],
        'gamma': [0, 0.1 ],
        'reg_alpha': [0, 1e-5,1e-4, 1e-3],
        'reg_lambda': [0, 1e-5, 1e-3,1e-4,1e-2],
        # 'gpu_id ':[0]
    }
    # param_grid = {
    #     'max_depth': [3, ],
    #     'learning_rate': [0.01 ],
    #     # "grow_policy" :[0,1] ,
    #     'n_estimators': [50 ],
    #     'subsample': [0.8 ],
    #     'colsample_bytree': [0.8 ],
    #     'gamma': [  0.2],
    #     'reg_alpha': [0, 1e-5, 1e-3],
    #     # 'reg_lambda': [0, 1e-5, 1e-3]
    # }
    ua = 'ua' if usearea else 'na'
    un = 'un' if usenorm else 'nn'
    basename = f'XGBR_{ua}_{un}_{tag}' 
    print(f'basename {basename}')
    timestamp =  time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp ))
    # run(path,model=model,param_grid=param_grid,metircs=metircs,usearea=usearea,namefile=namefile)


    X_train, X_test, y_train, y_test = dataprocess(path=path,metircs=metircs,usearea=usearea,usenorm=usenorm)
    data=[X_train, X_test, y_train, y_test ]

        #保存数据  
    print('saving data ...')

    datapath  = f'/home/dls/data/openmmlab/test_RandomForestRegressor/{basename}_{timestamp} '
    print(f'datapath {datapath}')
    savedata(data=data,datapath=datapath)


    print('running...')
    grid_search ,r2 ,rmse,best_rf = run(model=model,param_grid=param_grid,data=data )

    modelpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/bestmodel'
    os.makedirs(modelpath,exist_ok=True)


    # 输出log
    logpath =  '/home/dls/data/openmmlab/test_RandomForestRegressor/log'
    logname =   os.path.join(logpath,f'{basename}.log') 
    os.makedirs(logpath,exist_ok=True)
    with open(logname,'a+') as f :
        strval = f'Best parameters:  {grid_search.best_params_} \n'
        strval1 = 'R2 score: {:.2f}\n'.format(r2)
        strval2 = 'RMSE: {:.2f}\n'.format(rmse)
        f.write(strval)
        f.write(strval1)
        f.write(strval2)
        f.write('\n')
        f.write(str(X_train.columns))
        f.write('\n')
        f.write(str(y_train.name))
        f.write('\n')
        f.write(f' use norm {usenorm} ')
        f.write('\n')
        f.write(f' metircs {metircs} ')
        f.write('\n')
        f.write(f' use area {usearea} ')

    # 输出R2和RMSE
    print(strval)
    print(str(X_train.columns))
    print(str(y_train.name))
    print('R2 score: {:.2f}'.format(r2))
    print('RMSE: {:.2f}'.format(rmse))

    # 保存grid search
    model_name = basename+f'_{round(r2,3)}_{timestamp}.pkl'
    gs_name = basename+f'_gs_{round(r2,3)}_{timestamp}.pkl'

    model_name = os.path.join(modelpath, model_name)
    gs_name = os.path.join(modelpath, gs_name)
    print(f'model_name {model_name}')
    print(f'gs_name {gs_name}')

    save_grid_search(best_rf,model_name)
    save_grid_search(grid_search,gs_name)



# 改变数据的读取方式 在计算相关性的时候输出文件 直接指定y 和x 
def mainxboost2(datapath,tag ,savemodelpath ):
    import xgboost as xgb
    # 设置 XGBoost 模型的参数空间
    # define XGBoost model

    param_grid = {
        'learning_rate': [ 0.001,  0.0001 ],
        # 'max_depth': [3,  5,None ],
        # 'booster':['gbtree','gblinear'],
        'n_estimators': [   8000 ],
        'subsample': [ 0.6, 0.8, 1.0],
        'colsample_bytree': [ 0.4,0.5,0.6 ] ,
        'gamma': [0.001,0.1],
        'reg_alpha': [0,   1e-2,1,5],
        'reg_lambda': [0,  1e-2,1,5],
        # 'gpu_id ':[0]
    }
    # param_grid = {
    #     'max_depth': [ 5  ],
    #     'learning_rate': [0.001  ],
    #     # 'booster':['gbtree','gblinear'],
    #     'n_estimators': [ 150  ],
    #     'subsample': [ 0.8 ],
    #     'colsample_bytree': [ 0.8 ],
    #     'gamma': [ 0.1 ],
    #     'reg_alpha': [  1e-3],
    #     'reg_lambda': [ 1e-2],
    #     # 'gpu_id ':[0]
    # }
    # param_grid = {
    #     'max_depth': [3, ],
    #     'learning_rate': [0.01 ],
    #     # "grow_policy" :[0,1] ,
    #     'n_estimators': [50 ],
    #     'subsample': [0.8 ],
    #     'colsample_bytree': [0.8 ],
    #     'gamma': [  0.2],
    #     'reg_alpha': [0, 1e-5, 1e-3],
    #     # 'reg_lambda': [0, 1e-5, 1e-3]
    # }
  
    basename = f'XGBR_{tag}' 
    print(f'basename {basename}')
    timestamp =  time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp ))
    # data = pd.read_csv(datapath)
    # 对数据进行处理  根据相关性 筛选特征 并进行归一化
    data = dataprocessdata_rdp(datapath)
    
    namex = data.columns[:-1]
    namey = data.columns[-1]
 
    # exit()

    datax = data[namex]
    datay = data[namey]
    # print('namex',datax_filter.columns)
    print('namey',namey)
    print('namex',namex)
    # exit()
    X_train, X_test, y_train, y_test = train_test_split(datax,datay, test_size=0.2, random_state=randomseed)
    # X_train, X_test, y_train, y_test = dataprocess(path=path,metircs=metircs,usearea=usearea,usenorm=usenorm)
    data=[X_train, X_test, y_train, y_test ]

    def fit_model(params):
        result = {}
        model = xgb.XGBRegressor(**params,random_state=randomseed)
        model.fit(X_train, y_train )
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        result['r2acc'] = accuracy

        y_predt = model.predict(X_train)
        accuracyt = r2_score(y_train, y_predt)
        result['train_r2acc'] = accuracyt
        for key, value in params.items():
            result[key] = value
        print(result)
        return result
    def parallelrun():
        # 创建参数列表
        param_list = list(ParameterGrid(param_grid))

        # 使用 joblib 并行执行函数
        # results = Parallel(n_jobs=16, verbose=0)(delayed(fit_model)(params) for params in tqdm(param_list))
        results = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(params) for params in param_list)
        df= pd.DataFrame(results)
        df.to_csv(f'{basename}_{timestamp}_XGBRresults.csv',index=False)
        print(df)
        # 获取 age 列最大值所在的行
        max_age_index = df['r2acc'].idxmax()
        max_age_row = df.loc[max_age_index]

        # 打印结果
        print(max_age_row)
    
    parallelrun()
        #保存数据  
    # print('saving data ...')

    # savedatapath  = f'/home/dls/data/openmmlab/test_RandomForestRegressor/{basename}_{timestamp}_data'
    # print(f'datapath {datapath}')
    # savedata(data=data,datapath=savedatapath)

    # print('running...')
    # grid_search ,r2 ,rmse,best_rf = run(model=model,param_grid=param_grid,data=data )
    # # 输出log
    # logpath =  '/home/dls/data/openmmlab/test_RandomForestRegressor/log'
    # logname =   os.path.join(logpath,f'{basename}.log') 
    # os.makedirs(logpath,exist_ok=True)
    # with open(logname,'a+') as f :
    #     f.write('\n')
    #     f.write(timestamp)
    #     strval = f'Best parameters:  {grid_search.best_params_} \n'
    #     strval1 = 'R2 score: {:.2f}\n'.format(r2)
    #     strval2 = 'RMSE: {:.2f}\n'.format(rmse)
    #     f.write(strval)
    #     f.write(strval1)
    #     f.write(strval2)
    #     f.write('\n')
    #     f.write(str(X_train.columns))
    #     f.write('\n')
    #     f.write(str(y_train.name))
    #     f.write('\n')
    #     f.write(f'{param_grid}')

    # # 输出R2和RMSE
    # print(strval)
    # print(str(X_train.columns))
    # print(str(y_train.name))
    # print('R2 score: {:.2f}'.format(r2))
    # print('RMSE: {:.2f}'.format(rmse))

    
    # # 保存grid search
    # model_name = basename+f'_{round(r2,3)}_{timestamp}.pkl'
    # gs_name = basename+f'_gs_{round(r2,3)}_{timestamp}.pkl'

    # os.makedirs(savemodelpath,exist_ok=True)
    # model_name = os.path.join(savemodelpath, model_name)
    # gs_name = os.path.join(savemodelpath, gs_name)
    # print(f'save model_name {model_name}')
    # print(f'save gs_name {gs_name}')

    # save_grid_search(best_rf,model_name)
    # save_grid_search(grid_search,gs_name)

    # y_pred = best_rf.predict(X_test)
    
    # # 绘制训练结果和测试结果的图表
    # plt.scatter(y_train, best_rf.predict(X_train), label='Train')
    # plt.scatter(y_test, y_pred, label='Test')
    # plt.plot([0.5, 1], [0.5, 1], '--k', label='Ideal')
    # plt.xlabel('True Values')
    # plt.ylabel('Predictions')
    # plt.legend()
    # picpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/pic'
    # plt.savefig(os.path.join(picpath,f'{basename}.png'))

def mainGradientBoostingRegressor(usearea,datapath,tag ,savemodelpath ):
    model =  GradientBoostingRegressor(random_state=randomseed)
    # 定义参数空间 把早停放进去
    # param_grid = {
    #     'learning_rate': [0.05, 0.01,0.02],
    #     'n_estimators': [ 100,500,1000,2000,5000],
    #     'max_depth': [3, 4, 5,6,7,8],
    #     'min_samples_split':[ 10,50,100],
    #     # 'n_iter_no_change': [5, 10, None]
    # }
    param_grid = {
        'learning_rate': [  0.004, 0.005],
        'n_estimators': [   2000,2200 ],
        'loss':['squared_error', 'absolute_error', 'huber'],
        'max_depth': [  6],
        'min_samples_split':[   0.62 ,0.61,0.6 ],
        'min_samples_leaf':[   0.01,0.05 ],
        'warm_start':[True ],
        'max_features' : [ 'sqrt' ],
        # 'min_samples_leaf':[20],
        'subsample':[1.0  ],
        # 'n_iter_no_change': [5, 10, None]
    }
    # 梯度提升树
    # GradientBoostingRegressor是一种基于决策树的梯度提升算法，可以用于回归问题。下面是一些可能需要调整的重要参数：
    # n_estimators：弱学习器的数量（即决策树的数量）。默认值为100，增加该值通常可以提高模型的准确性，但也会增加计算成本。
    # learning_rate：每个决策树的权重缩减系数。默认值为0.1，通常应该在0.01至0.2之间。较小的学习率会使模型收敛速度变慢，但可以提高泛化能力；较大的学习率会使模型更快地收敛，但可能导致过拟合。
    # max_depth：决策树的最大深度。默认值为3，通常应该在2至8之间。增加最大深度可以提高模型的准确性，但也可能导致过拟合。
    # min_samples_split ：决策树的最小样本分割数量。默认值为2，通常应该在100至1000之间。增加该值可以减少过拟合。
    # min_samples_leaf ：决策树的最小叶节点样本数量。默认值为1，通常应该在10至100之间。增加该值可以减少过拟合。
    # subsample ： 每个决策树的样本采样比例。默认值为1，表示使用所有样本进行训练。减小该值可以减少过拟合。
    ua = 'ua' if usearea else 'na'
    basename = f'GBR_{ua}{tag}' 
    print(f'basename {basename}')
    timestamp =  time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp ))
    # run(path,model=model,param_grid=param_grid,metircs=metircs,usearea=usearea,namefile=namefile)

    data = pd.read_csv(datapath)
    namex = data.columns[:-1]
    namey = data.columns[-1]
    # 根据相关性 FULLER4D_5_rel_nocv_maxscaler，筛选特征
    datax = data[namex]
    datay = data[namey]
    dropcolumns =['angle_std_maxscaler','angle_maxmin_std_maxscaler','area_mean_maxscaler','area_std_maxscaler','edge_dis_maxmin_std_maxscaler','frac_std_maxscaler']
    namex_filter=[name for name in  datax.columns if name not in dropcolumns ]
    # print('namex',namex_filter )

    datax_filter = datax[namex_filter]
    # datax_filter =datax_filter.loc[:,~datax_filter.columns.str.contains('_maxmin')]
    # datax_filter =datax_filter.loc[:,~datax_filter.columns.str.contains('_std')]
    # datax_filter =datax_filter.loc[:,~datax_filter.columns.str.contains('per')]


    print('datax_filter',datax_filter.columns)
    # print('namey',namey)
    # exit()
    X_train, X_test, y_train, y_test = train_test_split(datax_filter,datay, test_size=0.2, random_state=randomseed)
    # X_train, X_test, y_train, y_test = dataprocess(path=path,metircs=metircs,usearea=usearea,usenorm=usenorm)
    data=[X_train, X_test, y_train, y_test ]
        #保存数据  
    # print('saving data ...')

    # savedatapath  = f'/home/dls/data/openmmlab/test_RandomForestRegressor/{basename}_{timestamp}_data'
    # print(f'datapath {datapath}')
    # savedata(data=data,datapath=savedatapath)


    print('running...')
    grid_search ,r2 ,rmse,best_rf = run(model=model,param_grid=param_grid,data=data )
    # 输出log
    logpath =  '/home/dls/data/openmmlab/test_RandomForestRegressor/log'
    logname =   os.path.join(logpath,f'{basename}.log') 
    os.makedirs(logpath,exist_ok=True)
    with open(logname,'a+') as f :
        
        f.write('\n')
        f.write(timestamp)

        strval = f'Best parameters:  {grid_search.best_params_} \n'
        strval1 = 'R2 score: {:.2f}\n'.format(r2)
        strval2 = 'RMSE: {:.2f}\n'.format(rmse)
        f.write(strval)
        f.write(strval1)
        f.write(strval2)
        f.write('\n')
        f.write(str(X_train.columns))
        f.write('\n')
        f.write(str(y_train.name))
        f.write('\n')
        f.write(f' use area {usearea} ')
        f.write('\n')
        f.write(f'{param_grid}')

    # 输出R2和RMSE
    print(strval)
    print(str(X_train.columns))
    print(str(y_train.name))
    print('R2 score: {:.2f}'.format(r2))
    print('RMSE: {:.2f}'.format(rmse))

    # # 保存grid search
    model_name = basename+f'_{round(r2,3)}_{timestamp}.pkl'
    gs_name = basename+f'_gs_{round(r2,3)}_{timestamp}.pkl'

    os.makedirs(savemodelpath,exist_ok=True)
    model_name = os.path.join(savemodelpath, model_name)
    gs_name = os.path.join(savemodelpath, gs_name)
    print(f'save model_name {model_name}')
    print(f'save gs_name {gs_name}')

    save_grid_search(best_rf,model_name)
    save_grid_search(grid_search,gs_name)

    # y_pred = best_rf.predict(X_test)
    
    # # 绘制训练结果和测试结果的图表
    # plt.scatter(y_train, best_rf.predict(X_train), label='Train')
    # plt.scatter(y_test, y_pred, label='Test')
    # plt.plot([0.5, 1], [0.5, 1], '--k', label='Ideal')
    # plt.xlabel('True Values')
    # plt.ylabel('Predictions')
    # plt.legend()
    # picpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/pic'
    # plt.savefig(os.path.join(picpath,f'{basename}.png'))


def mainGradientBoostingRegressor2(datapath,tag ,savemodelpath):

    # 定义参数空间 把早停放进去
    # param_grid = {
    #     'learning_rate': [0.05, 0.01,0.02],
    #     'n_estimators': [ 100,500,1000,2000,5000],
    #     'max_depth': [3, 4, 5,6,7,8],
    #     'min_samples_split':[ 10,50,100],
    #     # 'n_iter_no_change': [5, 10, None]
    # }
    # param_grid = {
    #     'learning_rate': [ 0.0001,0.001,0.01,0.1],
    #     'n_estimators': [    500,1000,2000,5000,8000 ],
    #     'loss':['squared_error', 'absolute_error', 'huber'],
    #     'max_depth': [  6],
    #     'min_samples_split':[0.0001,0.001,0.01,0.1],
    #     'min_samples_leaf':[   0.0001,0.001,0.01,0.1],
    #     'warm_start':[True ],
    #     # 'max_features' : [ 'sqrt' ],
    #     # 'min_samples_leaf':[20],
    #     'subsample':[1.0,0.6,0.8  ],
    #     # 'n_iter_no_change': [5, 10, None]
    # }
    param_grid = {
        'learning_rate': [ 0.0001,0.0005,0.001 ],
        'n_estimators': [    5000,8000 ],
        'loss':[ 'huber'],
        'max_depth': [ 4, 6, 8],
        'min_samples_split':[  1e-5,  0.0001, 0.001 ],
        'min_samples_leaf': [  1e-5,  0.0001, 0.001 ],
        'warm_start':[True ],
        # 'max_features' : [ 'sqrt' ],
        # 'min_samples_leaf':[20],
        'subsample':[1.0,0.9 ],
        # 'n_iter_no_change': [5, 10, None]
    }
    # 梯度提升树
    # GradientBoostingRegressor是一种基于决策树的梯度提升算法，可以用于回归问题。下面是一些可能需要调整的重要参数：
    # n_estimators：弱学习器的数量（即决策树的数量）。默认值为100，增加该值通常可以提高模型的准确性，但也会增加计算成本。
    # learning_rate：每个决策树的权重缩减系数。默认值为0.1，通常应该在0.01至0.2之间。较小的学习率会使模型收敛速度变慢，但可以提高泛化能力；较大的学习率会使模型更快地收敛，但可能导致过拟合。
    # max_depth：决策树的最大深度。默认值为3，通常应该在2至8之间。增加最大深度可以提高模型的准确性，但也可能导致过拟合。
    # min_samples_split ：决策树的最小样本分割数量。默认值为2，通常应该在100至1000之间。增加该值可以减少过拟合。
    # min_samples_leaf ：决策树的最小叶节点样本数量。默认值为1，通常应该在10至100之间。增加该值可以减少过拟合。
    # subsample ： 每个决策树的样本采样比例。默认值为1，表示使用所有样本进行训练。减小该值可以减少过拟合。
    basename = f'GBR_2_{tag}' 
    print(f'basename {basename}')
    timestamp =  time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp ))
    # data = pd.read_csv(datapath)
    # 对数据进行处理  根据相关性 筛选特征 并进行归一化
    data = dataprocessdata_rdp(datapath)
    
    namex = data.columns[:-1]
    namey = data.columns[-1]
 
    # exit()

    datax = data[namex]
    datay = data[namey]
    # print('namex',datax_filter.columns)
    print('namey',namey)
    print('namex',namex)
    # exit()
    X_train, X_test, y_train, y_test = train_test_split(datax,datay, test_size=0.1, random_state=randomseed)
    # X_train, X_test, y_train, y_test = dataprocess(path=path,metircs=metircs,usearea=usearea,usenorm=usenorm)
    data=[X_train, X_test, y_train, y_test ]
    print('saving data ...')

    savedatapath  = f'/home/dls/data/openmmlab/test_RandomForestRegressor/traindata/{basename}_{timestamp}_data'
    print(f'datapath {savedatapath}')
    savedata(data=data,datapath=savedatapath)
    def fit_model(params):
        result = {}
        model =  GradientBoostingRegressor(**params,random_state=randomseed)
        model.fit(X_train, y_train )
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        result['r2acc'] = accuracy

        y_predt = model.predict(X_train)
        accuracyt = r2_score(y_train, y_predt)
        result['train_r2acc'] = accuracyt
        for key, value in params.items():
            result[key] = value
        # print(result)
        return result
    def parallelrun():
        # 创建参数列表
        param_list = list(ParameterGrid(param_grid))

        # 使用 joblib 并行执行函数
        results = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(params) for params in tqdm(param_list))
        # results = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(params) for params in param_list)
        df= pd.DataFrame(results)
        # df.to_csv(f'{basename}_{timestamp}_GBRresults.csv',index=False)
        print(df)
        # 获取 age 列最大值所在的行
        max_age_index = df['r2acc'].idxmax()
        max_age_row = df.loc[max_age_index]
        value = df['r2acc'].max()
        df.to_csv(f'{basename}_{timestamp}_{value}_GBRresults.csv',index=False)
        # 打印结果
        print(max_age_row)
    
    parallelrun()

# y值不进行归一化
def mainGradientBoostingRegressor3(datapath,tag ,savemodelpath):

    # 定义参数空间 把早停放进去
    # param_grid = {
    #     'learning_rate': [0.05, 0.01,0.02],
    #     'n_estimators': [ 100,500,1000,2000,5000],
    #     'max_depth': [3, 4, 5,6,7,8],
    #     'min_samples_split':[ 10,50,100],
    #     # 'n_iter_no_change': [5, 10, None]
    # }
    # param_grid = {
    #     'learning_rate': [ 0.0001,0.001,0.01,0.1],
    #     'n_estimators': [    500,1000,2000,5000,8000 ],
    #     'loss':['squared_error', 'absolute_error', 'huber'],
    #     'max_depth': [  6],
    #     'min_samples_split':[0.0001,0.001,0.01,0.1],
    #     'min_samples_leaf':[   0.0001,0.001,0.01,0.1],
    #     'warm_start':[True ],
    #     # 'max_features' : [ 'sqrt' ],
    #     # 'min_samples_leaf':[20],
    #     'subsample':[1.0,0.6,0.8  ],
    #     # 'n_iter_no_change': [5, 10, None]
    # }    'learning_rate': [  1e-5,5e-4, 6e-4,7e-4,8e-4 , 9e-4,],
        # 'n_estimators': [     5000,6000 8000,4000]
    param_grid = {
        'learning_rate':[ 3e-3,4e-3 ],
        'n_estimators':[  8000, 9000 ],
        'loss':['absolute_error'],
        'max_depth':[ 10],
        'min_samples_split':[1e-07, 1e-08, 5e-07 ],
        'min_samples_leaf':[1e-07, 1e-08, 5e-07 ],
        'warm_start':[True],
        'subsample':[ 0.7, 0.8],
        'ccp_alpha':[0,0.008]
    }
        # 输出log

    # param_grid = {

    # 'learning_rate':[0.0007, 0.00075],
    # 'n_estimators':[9000 ],
    # 'loss':['huber'],
    # 'max_depth':[ 2 ],
    # 'min_samples_split':[1e-07, 3e-07, 5e-07],
    # 'min_samples_leaf':[1e-07, 3e-07, 5e-07],
    # 'warm_start':[True],
    # 'subsample':[1.0 ],
    # }
    # 梯度提升树
    # GradientBoostingRegressor是一种基于决策树的梯度提升算法，可以用于回归问题。下面是一些可能需要调整的重要参数：
    # n_estimators：弱学习器的数量（即决策树的数量）。默认值为100，增加该值通常可以提高模型的准确性，但也会增加计算成本。
    # learning_rate：每个决策树的权重缩减系数。默认值为0.1，通常应该在0.01至0.2之间。较小的学习率会使模型收敛速度变慢，但可以提高泛化能力；较大的学习率会使模型更快地收敛，但可能导致过拟合。
    # max_depth：决策树的最大深度。默认值为3，通常应该在2至8之间。增加最大深度可以提高模型的准确性，但也可能导致过拟合。
    # min_samples_split ：决策树的最小样本分割数量。默认值为2，通常应该在100至1000之间。增加该值可以减少过拟合。
    # min_samples_leaf ：决策树的最小叶节点样本数量。默认值为1，通常应该在10至100之间。增加该值可以减少过拟合。
    # subsample ： 每个决策树的样本采样比例。默认值为1，表示使用所有样本进行训练。减小该值可以减少过拟合。
    basename = f'GBR_3_{tag}' 
    print(f'basename {basename}')
    timestamp =  time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp ))
    # data = pd.read_csv(datapath)
    # 对数据进行处理  根据相关性 筛选特征 并进行归一化
    data = dataprocessdata_rdp_normx(datapath)
    logpath =  '/home/dls/data/openmmlab/test_RandomForestRegressor/log'
    logname =   os.path.join(logpath,f'{basename}.log') 
    os.makedirs(logpath,exist_ok=True)
    with open(logname,'a+') as f :
        f.write(f'\n')
        f.write(f'{basename}\n')

        for key ,value in param_grid.items():
            f.write(f'{key}:{value}\n')
    namex = data.columns[:-1]
    namey = data.columns[-1]
 
    datax = data[namex]
    datay = data[namey]
    # print('namex',datax_filter.columns)
    print('namey',namey)
    print('namex',namex)
    # exit()
    X_train, X_test, y_train, y_test = train_test_split(datax,datay, test_size=0.1, random_state=randomseed)
    # X_train, X_test, y_train, y_test = dataprocess(path=path,metircs=metircs,usearea=usearea,usenorm=usenorm)
    data=[X_train, X_test, y_train, y_test ]
    print('saving data ...')


    def fit_model(params):
        result = {}
        model =  GradientBoostingRegressor(**params,random_state=randomseed)
        model.fit(X_train, y_train )
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        result['r2acc'] = accuracy

        y_predt = model.predict(X_train)
        accuracyt = r2_score(y_train, y_predt)
        result['train_r2acc'] = accuracyt
        for key, value in params.items():
            result[key] = value
        # print(result)
        return (result,model)
    def parallelrun():
        # 创建参数列表
        param_list = list(ParameterGrid(param_grid))

        # 使用 joblib 并行执行函数
        results  = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(params) for params in tqdm(param_list))
        # results = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(params) for params in param_list)
        resultrcc = [result[0] for result in results]
        models = [result[1] for result in results]
        df= pd.DataFrame(resultrcc)
        # df.to_csv(f'{basename}_{timestamp}_GBRresults.csv',index=False)
        print(df)
        # 获取 age 列最大值所在的行
        max_age_index = df['r2acc'].idxmax()
        max_age_row = df.loc[max_age_index]
        model =models[max_age_index]
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_test)
        s1= pd.Series(y_test,name='y_test')
        s2= pd.Series(y_pred,name='y_pred')
        df_pre = pd.concat([s1, s2], axis=1)
        value = df['r2acc'].max()
        df.to_csv(f'{basename}_{timestamp}_{value}_GBRresults.csv',index=False)
        df_pre.to_csv(f'{basename}_{timestamp}_{value}_GBRpreresults.csv',index=False)

        # 打印结果
        print(max_age_row)
        with open(logname,'a+') as f :
               max_age_row.to_csv(f,  index=False)
        model_name = basename+f'_{round(value,3)}_{timestamp}.pkl'
        model_name = os.path.join(savemodelpath, model_name)
 
        print(f'model_name {model_name}')
        save_grid_search(model,model_name)
        savedatapath  = f'/home/dls/data/openmmlab/test_RandomForestRegressor/traindata/{basename}_{timestamp}_{value}_data'
        print(f'datapath {savedatapath}')
        savedata(data=data,datapath=savedatapath)
    parallelrun()

#x  y值都不进行归一化
def mainGradientBoostingRegressor4(datapath,tag ,savemodelpath):

    # 定义参数空间 把早停放进去
    # param_grid = {
    #     'learning_rate': [0.05, 0.01,0.02],
    #     'n_estimators': [ 100,500,1000,2000,5000],
    #     'max_depth': [3, 4, 5,6,7,8],
    #     'min_samples_split':[ 10,50,100],
    #     # 'n_iter_no_change': [5, 10, None]
    # }
    # param_grid = {
    #     'learning_rate': [ 0.0001,0.001,0.01,0.1],
    #     'n_estimators': [    500,1000,2000,5000,8000 ],
    #     'loss':['squared_error', 'absolute_error', 'huber'],
    #     'max_depth': [  6],
    #     'min_samples_split':[0.0001,0.001,0.01,0.1],
    #     'min_samples_leaf':[   0.0001,0.001,0.01,0.1],
    #     'warm_start':[True ],
    #     # 'max_features' : [ 'sqrt' ],
    #     # 'min_samples_leaf':[20],
    #     'subsample':[1.0,0.6,0.8  ],
    #     # 'n_iter_no_change': [5, 10, None]
    # }    'learning_rate': [  1e-5,5e-4, 6e-4,7e-4,8e-4 , 9e-4,],
        # 'n_estimators': [     5000,6000 8000,4000]
    param_grid = {
        'learning_rate':[ 1e-4,3e-3,5e-3 ],
        'n_estimators':[  5000 ],
        'loss':['absolute_error'],
        'max_depth':[  8,10],
        'min_samples_split':[10 ],
        'min_samples_leaf':[10 ],
        'warm_start':[True],
        'subsample':[ 0.6,  0.8],
        'ccp_alpha':[0.01,0.005  ]
    }
        # 输出log

    # param_grid = {

    # 'learning_rate':[0.0007, 0.00075],
    # 'n_estimators':[9000 ],
    # 'loss':['huber'],
    # 'max_depth':[ 2 ],
    # 'min_samples_split':[1e-07, 3e-07, 5e-07],
    # 'min_samples_leaf':[1e-07, 3e-07, 5e-07],
    # 'warm_start':[True],
    # 'subsample':[1.0 ],
    # }
    # 梯度提升树
    # GradientBoostingRegressor是一种基于决策树的梯度提升算法，可以用于回归问题。下面是一些可能需要调整的重要参数：
    # n_estimators：弱学习器的数量（即决策树的数量）。默认值为100，增加该值通常可以提高模型的准确性，但也会增加计算成本。
    # learning_rate：每个决策树的权重缩减系数。默认值为0.1，通常应该在0.01至0.2之间。较小的学习率会使模型收敛速度变慢，但可以提高泛化能力；较大的学习率会使模型更快地收敛，但可能导致过拟合。
    # max_depth：决策树的最大深度。默认值为3，通常应该在2至8之间。增加最大深度可以提高模型的准确性，但也可能导致过拟合。
    # min_samples_split ：决策树的最小样本分割数量。默认值为2，通常应该在100至1000之间。增加该值可以减少过拟合。
    # min_samples_leaf ：决策树的最小叶节点样本数量。默认值为1，通常应该在10至100之间。增加该值可以减少过拟合。
    # subsample ： 每个决策树的样本采样比例。默认值为1，表示使用所有样本进行训练。减小该值可以减少过拟合。
    basename = f'GBR_4_{tag}' 
    print(f'basename {basename}')
    timestamp =  time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp ))
    # data = pd.read_csv(datapath)
    # 对数据进行处理  根据相关性 筛选特征 并进行归一化
    data = dataprocessdata_rdp_nonorm(datapath)
    logpath =  '/home/dls/data/openmmlab/test_RandomForestRegressor/log'
    logname =   os.path.join(logpath,f'{basename}.log') 
    os.makedirs(logpath,exist_ok=True)
    with open(logname,'a+') as f :
        f.write(f'\n')
        f.write(f'{basename}\n')

        for key ,value in param_grid.items():
            f.write(f'{key}:{value}\n')
    namex = data.columns[:-1]
    namey = data.columns[-1]
 
    datax = data[namex]
    datay = data[namey]
    # print('namex',datax_filter.columns)
    print('namey',namey)
    print('namex',namex)
    # exit()
    X_train, X_test, y_train, y_test = train_test_split(datax,datay, test_size=0.1, random_state=randomseed)
    # X_train, X_test, y_train, y_test = dataprocess(path=path,metircs=metircs,usearea=usearea,usenorm=usenorm)
    data=[X_train, X_test, y_train, y_test ]
    print('saving data ...')


    def fit_model(params):
        result = {}
        model =  GradientBoostingRegressor(**params,random_state=randomseed)
        model.fit(X_train, y_train )
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        result['r2acc'] = accuracy

        y_predt = model.predict(X_train)
        accuracyt = r2_score(y_train, y_predt)
        result['train_r2acc'] = accuracyt
        for key, value in params.items():
            result[key] = value
        # print(result)
        return (result,model)
    def parallelrun():
        # 创建参数列表
        param_list = list(ParameterGrid(param_grid))

        # 使用 joblib 并行执行函数
        results  = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(params) for params in tqdm(param_list))
        # results = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(params) for params in param_list)
        resultrcc = [result[0] for result in results]
        models = [result[1] for result in results]
        df= pd.DataFrame(resultrcc)
        # df.to_csv(f'{basename}_{timestamp}_GBRresults.csv',index=False)
        print(df)
        # 获取 age 列最大值所在的行
        max_age_index = df['r2acc'].idxmax()
        max_age_row = df.loc[max_age_index]
        model =models[max_age_index]
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        s1= pd.Series(y_pred,name='y_pred')
        s2= pd.Series(y_pred_train,name='y_pred_train')
        df_pre = pd.concat([s1, s2], axis=1)
        value = df['r2acc'].max()
        df.to_csv(f'{basename}_{timestamp}_{value}_GBRresults.csv',index=False)
        df_pre.to_csv(f'{basename}_{timestamp}_{value}_GBRpreresults.csv',index=False)

        # 打印结果
        print(max_age_row)
        with open(logname,'a+') as f :
               max_age_row.to_csv(f,  index=False)
        model_name = basename+f'_{round(value,3)}_{timestamp}.pkl'
        model_name = os.path.join(savemodelpath, model_name)
 
        print(f'model_name {model_name}')
        save_grid_search(model,model_name)
        savedatapath  = f'/home/dls/data/openmmlab/test_RandomForestRegressor/traindata/{basename}_{timestamp}_{value}_data'
        print(f'datapath {savedatapath}')
        savedata(data=data,datapath=savedatapath)
    parallelrun()



def mainGradientBoostingRegressorpca(datapath,tag ,savemodelpath):

    # 定义参数空间 把早停放进去
    # param_grid = {
    #     'learning_rate': [0.05, 0.01,0.02],
    #     'n_estimators': [ 100,500,1000,2000,5000],
    #     'max_depth': [3, 4, 5,6,7,8],
    #     'min_samples_split':[ 10,50,100],
    #     # 'n_iter_no_change': [5, 10, None]
    # }
    param_grid = {
        'learning_rate': [ 0.0001,0.001,0.01,0.1],
        'n_estimators': [    500,1000,2000,5000,8000 ],
        'loss':['squared_error', 'absolute_error', 'huber'],
        'max_depth': [  6],
        'min_samples_split':[0.0001,0.001,0.01,0.1],
        'min_samples_leaf':[   0.0001,0.001,0.01,0.1],
        'warm_start':[True ],
        # 'max_features' : [ 'sqrt' ],
        # 'min_samples_leaf':[20],
        'subsample':[1.0,0.6,0.8  ],
        # 'n_iter_no_change': [5, 10, None]
    }

    # 梯度提升树
    # GradientBoostingRegressor是一种基于决策树的梯度提升算法，可以用于回归问题。下面是一些可能需要调整的重要参数：
    # n_estimators：弱学习器的数量（即决策树的数量）。默认值为100，增加该值通常可以提高模型的准确性，但也会增加计算成本。
    # learning_rate：每个决策树的权重缩减系数。默认值为0.1，通常应该在0.01至0.2之间。较小的学习率会使模型收敛速度变慢，但可以提高泛化能力；较大的学习率会使模型更快地收敛，但可能导致过拟合。
    # max_depth：决策树的最大深度。默认值为3，通常应该在2至8之间。增加最大深度可以提高模型的准确性，但也可能导致过拟合。
    # min_samples_split ：决策树的最小样本分割数量。默认值为2，通常应该在100至1000之间。增加该值可以减少过拟合。
    # min_samples_leaf ：决策树的最小叶节点样本数量。默认值为1，通常应该在10至100之间。增加该值可以减少过拟合。
    # subsample ： 每个决策树的样本采样比例。默认值为1，表示使用所有样本进行训练。减小该值可以减少过拟合。
    basename = f'GBRpca_{tag}' 
    print(f'basename {basename}')
    timestamp =  time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp ))
    # data = pd.read_csv(datapath)
    # 对数据进行处理  根据相关性 筛选特征 并进行归一化
    data = dataprocessdata_rdp(datapath)
    
    namex = data.columns[:-1]
    namey = data.columns[-1]
 
    # exit()

    datax = data[namex]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(datax)
    print('pca 总方差',pca.explained_variance_ratio_)
    datapac = pd.DataFrame(X_pca,columns= ['rc1','rc2'])
    datay = data[namey]
    # print('namex',datax_filter.columns)
    print('namey',namey)
    print('namex',namex)
    # exit()

    X_train, X_test, y_train, y_test = train_test_split(datapac,datay, test_size=0.1, random_state=randomseed)
    # X_train, X_test, y_train, y_test = dataprocess(path=path,metircs=metircs,usearea=usearea,usenorm=usenorm)
    data=[X_train, X_test, y_train, y_test ]
    savedatapath  = f'/home/dls/data/openmmlab/test_RandomForestRegressor/traindata/{basename}_{timestamp}_data'
    print(f'datapath {savedatapath}')
    savedata(data=data,datapath=savedatapath)
    def fit_model(params):
        result = {}
        model =  GradientBoostingRegressor(**params,random_state=randomseed)
        model.fit(X_train, y_train )
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        result['r2acc'] = accuracy

        y_predt = model.predict(X_train)
        accuracyt = r2_score(y_train, y_predt)
        result['train_r2acc'] = accuracyt
        for key, value in params.items():
            result[key] = value
        # print(result)
        return result
    def parallelrun():
        # 创建参数列表
        param_list = list(ParameterGrid(param_grid))

        # 使用 joblib 并行执行函数
        results = Parallel(n_jobs=16, verbose=0)(delayed(fit_model)(params) for params in tqdm(param_list))
        # results = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(params) for params in param_list)
        df= pd.DataFrame(results)
        print(df)
        # 获取 age 列最大值所在的行
        max_age_index = df['r2acc'].idxmax()
        max_age_row = df.loc[max_age_index]
        value = df['r2acc'].max()
        df.to_csv(f'{basename}_{timestamp}_{value}_GBRresults.csv',index=False)
        # 打印结果
        print(max_age_row)
    
    parallelrun()

import seaborn as sns
def mainGradientBoostingClassifier( datapath,tag ,savemodelpath ):
    param_grid = {
        'learning_rate': [  0.1, 0.01 ],
        'n_estimators': [    100, 2000,5000 ],
        'max_depth': [  3,6,20],
        'min_samples_split':[  0.0001, 0.01,0.5,2],
        'min_samples_leaf':[    0.0001, 0.01,0.5 ,3],
        'warm_start':[True ],
        # 'max_features' : [   'sqrt', 'log2',None  ],
        # 'min_samples_leaf':[20],
        'subsample':[1.0 ,0.8  ],
        # 'n_iter_no_change': [5, 10, None]
    }
    # param_grid = {
    #     'learning_rate': [  0.1,0.01 ,0.001],
    #     'n_estimators': [    500,2000],
    #     'max_depth': [  10],
    #     # 'min_samples_split':[    1.0 ],
    #     # 'min_samples_leaf':[      1.0  ],
    #     'warm_start':[True ],
    #     # 'max_features' : [  0.75  ],
    #     # 'min_samples_leaf':[20],
    #     # 'subsample':[ 0.9 ],
    #     # 'n_iter_no_change': [5, 10, None]
    # }
    basename = f'GBC_{tag}' 
    print(f'basename {basename}')
    timestamp =  time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp ))

    # data = pd.read_csv(datapath)
    # 对数据进行处理  根据相关性 筛选特征 并进行归一化
    data = dataprocessdata_rdp(datapath)
    
    namex = data.columns[:-1]
    namey = data.columns[-1]
    print('namex namey',namex,namey)
    # exit()

    datax = data[namex]
    datay = data[namey]
    nbins =20
    kbins = KBinsDiscretizer(n_bins=nbins, strategy='uniform',encode='ordinal')
    # kbins = KBinsDiscretizer(n_bins=10, strategy='uniform',encode='onehot')
    y_binned = kbins.fit_transform(datay.to_numpy() .reshape(-1, 1) ) .squeeze()
    data['bin'] = y_binned
    sns.histplot(datay, kde=True,bins=nbins )
    plt.savefig( f'{basename}_y_binned{nbins}.png')
    # print('y_binned',y_binned.shape )   
    # datax_filter =datax_filter.loc[:,~datax_filter.columns.str.contains('_maxmin')]
    # datax_filter =datax_filter.loc[:,~datax_filter.columns.str.contains('_std')]
    # datax_filter =datax_filter.loc[:,~datax_filter.columns.str.contains('per')]
    # print('namey',namey)
    # exit()
    # weight = compute_sample_weight('balanced', y_binned)
    X_train, X_test, y_train, y_test = train_test_split(datax,data['bin'], test_size=0.2, random_state=randomseed)
    data=[X_train, X_test, y_train, y_test ]
    savedatapath  = f'/home/dls/data/openmmlab/test_RandomForestRegressor/{basename}_{timestamp}_data'
    print(f'datapath {savedatapath}')
    savedata(data=data,datapath=savedatapath)
    exit()
    weight = compute_sample_weight('balanced', y_train)
    # weight = compute_class_weight(class_weight='balanced', classes= np.unique(y_train) , y=y_train.squeeze() )
    # results = []
    # for params in tqdm(ParameterGrid(param_grid)):
    #     result={}
    #     model = GradientBoostingClassifier(**params)
    #     # exit()
    #     model.fit(X_train,y_train, sample_weight=weight)
    #     # 对新数据进行预测
    #     y_pred_binned = model.predict(X_test)
    #     # print('y_pred_binned',y_pred_binned.shape )
    #     # print('y_test',y_test.shape )
    #     # accuracy =model.score(y_pred_binned.reshape(-1, 1), y_test.reshape(-1, 1))
    #     accuracy = accuracy_score(y_test, y_pred_binned)
    #     result['acc'] = accuracy
    #     for key ,value in params.items():
    #         result[key] =value
    #     results.append(result)
    # 定义函数
    def fit_model(params):
        result = {}
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train, sample_weight=weight)
        y_pred_binned = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_binned)
        result['acc'] = accuracy

        y_train_binned = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_binned)
        result['train_acc'] = train_accuracy
        for key, value in params.items():
            result[key] = value
        print(result)
        return result
    def parallelrun():
        # 创建参数列表
        param_list = list(ParameterGrid(param_grid))

        # 使用 joblib 并行执行函数
        results = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(params) for params in tqdm(param_list))
        # results = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(params) for params in param_list)
# 

        df= pd.DataFrame(results)
        df.to_csv(f'{basename}_{timestamp}_gbcresults.csv',index=False)
        print(df)
        
        # 获取 age 列最大值所在的行
        max_age_index = df['acc'].idxmax()
        max_age_row = df.loc[max_age_index]

        # 打印结果
        print(max_age_row)
    def testrun():
        results = []
        for params in tqdm(ParameterGrid(param_grid)):
            result={}
            model = GradientBoostingClassifier(**params)
            # exit()
            model.fit(X_train,y_train, sample_weight=weight)
            # 对新数据进行预测
            y_pred_binned = model.predict(X_test)
            # print('y_pred_binned',y_pred_binned.shape )
            # print('y_test',y_test.shape )
            # accuracy =model.score(y_pred_binned.reshape(-1, 1), y_test.reshape(-1, 1))
            accuracy = accuracy_score(y_test, y_pred_binned)
            result['acc'] = accuracy

            y_train_binned = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_binned)
            result['train_acc'] = train_accuracy

            for key ,value in params.items():
                result[key] =value
            results.append(result)
        df= pd.DataFrame(results)
        # df.to_csv(f'{basename}_{timestamp}_gbcresults.csv',index=False)
        print(df)
        # 获取 age 列最大值所在的行
        max_age_index = df['acc'].idxmax()
        max_age_row = df.loc[max_age_index]

        # 打印结果
        print(max_age_row)
    parallelrun()
    # grid_search ,acc , best_rf =runclass( model=model,param_grid=param_grid ,data=data)

    # # X_train, X_test, y_train, y_test = dataprocess(path=path,metircs=metircs,usearea=usearea,usenorm=usenorm)
    # # data=[X_train, X_test, y_train, y_test ]
    #     #保存数据  
    # # print('saving data ...')

    # # savedatapath  = f'/home/dls/data/openmmlab/test_RandomForestRegressor/{basename}_{timestamp}_data'
    # # print(f'datapath {datapath}')
    # # savedata(data=data,datapath=savedatapath)
    # # 进行因变量分箱
   
    # # # 保存grid search
    # model_name = basename+f'_{round(acc,3)}_{timestamp}.pkl'
    # gs_name = basename+f'_gs_{round(acc,3)}_{timestamp}.pkl'

    # os.makedirs(savemodelpath,exist_ok=True)
    # model_name = os.path.join(savemodelpath, model_name)
    # gs_name = os.path.join(savemodelpath, gs_name)
    # print(f'save model_name {model_name}')
    # print(f'save gs_name {gs_name}')

    # save_grid_search(best_rf,model_name)
    # save_grid_search(grid_search,gs_name)

    # # 训练随机森林分类模型
   
    # model.fit(X_train, y_train)



    # y_pred = best_rf.predict(X_test)
    
    # # 绘制训练结果和测试结果的图表
    # plt.scatter(y_train, best_rf.predict(X_train), label='Train')
    # plt.scatter(y_test, y_pred, label='Test')
    # plt.plot([0.5, 1], [0.5, 1], '--k', label='Ideal')
    # plt.xlabel('True Values')
    # plt.ylabel('Predictions')
    # plt.legend()
    # picpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/pic'
    # plt.savefig(os.path.join(picpath,f'{basename}.png'))

def mainxBoostingClassifier( datapath,tag ,savemodelpath ):
    # param_grid = {
    #     'learning_rate': [  0.1, 0.01,0.001,0.0001],
    #     'n_estimators': [    1000,1500,2000 ],
    #     'max_depth': [  3,6,10,100],
    #     'min_samples_split':[  0.0001, 0.01,0.5 ],
    #     'min_samples_leaf':[    0.0001, 0.01,0.5 ],
    #     'warm_start':[True ],
    #     'max_features' : [   'sqrt', 'log2',None  ],
    #     # 'min_samples_leaf':[20],
    #     'subsample':[1.0 ,0.8 ],
    #     # 'n_iter_no_change': [5, 10, None]
    # }
  
    param_grid = {
        'learning_rate': [ 0.001,  0.01, 0.1],
        'max_depth': [3, 7,10,None ],
        # 'booster':['gbtree','gblinear'],
        'n_estimators': [ 100,1000,5000 ],
        'subsample': [ 0.6, 0.8, 1.0],
        'colsample_bytree': [ 0.6, 0.8, 1.0] ,
        'gamma': [0,     0.5, 1 ],
        'reg_alpha': [0, 1e-4, 1e-2],
        'reg_lambda': [0, 1e-4, 1e-2],
        # 'gpu_id ':[0]
    }
    basename = f'XGBC_{tag}' 
    print(f'basename {basename}')
    timestamp =  time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp ))

    # data = pd.read_csv(datapath)
    # 对数据进行处理  根据相关性 筛选特征 并进行归一化
    data = dataprocessdata_rdp(datapath)
    
    namex = data.columns[:-1]
    namey = data.columns[-1]
    print('namex namey',namex,namey)
    # exit()

    datax = data[namex]
    datay = data[namey]
    nbins =10
    kbins = KBinsDiscretizer(n_bins=nbins, strategy='uniform',encode='ordinal')
    # kbins = KBinsDiscretizer(n_bins=10, strategy='uniform',encode='onehot')
    y_binned = kbins.fit_transform(datay.to_numpy() .reshape(-1, 1) ) .squeeze()
    sns.histplot(datay, kde=True,bins=nbins )
    plt.savefig( f'{basename}_y_binned{nbins}.png')
    # print('y_binned',y_binned.shape )   
    # datax_filter =datax_filter.loc[:,~datax_filter.columns.str.contains('_maxmin')]
    # datax_filter =datax_filter.loc[:,~datax_filter.columns.str.contains('_std')]
    # datax_filter =datax_filter.loc[:,~datax_filter.columns.str.contains('per')]
    # print('namey',namey)
    # exit()
    # weight = compute_sample_weight('balanced', y_binned)
    X_train, X_test, y_train, y_test = train_test_split(datax,y_binned, test_size=0.2, random_state=randomseed)
    data=[X_train, X_test, y_train, y_test ]
    weight = compute_sample_weight('balanced', y_train)
    # weight = compute_class_weight(class_weight='balanced', classes= np.unique(y_train) , y=y_train.squeeze() )
    # results = []
    # for params in tqdm(ParameterGrid(param_grid)):
    #     result={}
    #     model = GradientBoostingClassifier(**params)
    #     # exit()
    #     model.fit(X_train,y_train, sample_weight=weight)
    #     # 对新数据进行预测
    #     y_pred_binned = model.predict(X_test)
    #     # print('y_pred_binned',y_pred_binned.shape )
    #     # print('y_test',y_test.shape )
    #     # accuracy =model.score(y_pred_binned.reshape(-1, 1), y_test.reshape(-1, 1))
    #     accuracy = accuracy_score(y_test, y_pred_binned)
    #     result['acc'] = accuracy
    #     for key ,value in params.items():
    #         result[key] =value
    #     results.append(result)
    # 定义函数
    def fit_model(params):
        result = {}
        model = xgb.XGBClassifier(**params,random_state=randomseed)
        model.fit(X_train, y_train, sample_weight=weight)
        y_pred_binned = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_binned)
        result['acc'] = accuracy
        for key, value in params.items():
            result[key] = value
        # print(result)
        return result
    def parallelrun():
        # 创建参数列表
        param_list = list(ParameterGrid(param_grid))

        # 使用 joblib 并行执行函数
        results = Parallel(n_jobs=16, verbose=0)(delayed(fit_model)(params) for params in tqdm(param_list))
        # results = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(params) for params in param_list)
# 

        df= pd.DataFrame(results)
        df.to_csv(f'{basename}_{timestamp}_XGBCresults.csv',index=False)
        print(df)
        
        # 获取 age 列最大值所在的行
        max_age_index = df['acc'].idxmax()
        max_age_row = df.loc[max_age_index]

        # 打印结果
        print(max_age_row)
    def testrun():
        results = []
        for params in tqdm(ParameterGrid(param_grid)):
            result={}
            model = GradientBoostingClassifier(**params)
            # exit()
            model.fit(X_train,y_train, sample_weight=weight)
            # 对新数据进行预测
            y_pred_binned = model.predict(X_test)
            # print('y_pred_binned',y_pred_binned.shape )
            # print('y_test',y_test.shape )
            # accuracy =model.score(y_pred_binned.reshape(-1, 1), y_test.reshape(-1, 1))
            accuracy = accuracy_score(y_test, y_pred_binned)
            result['acc'] = accuracy
            
            for key ,value in params.items():
                result[key] =value
            results.append(result)
        df= pd.DataFrame(results)
        # df.to_csv(f'{basename}_{timestamp}_gbcresults.csv',index=False)
        print(df)
        # 获取 age 列最大值所在的行
        max_age_index = df['acc'].idxmax()
        max_age_row = df.loc[max_age_index]

        # 打印结果
        print(max_age_row)
    parallelrun()
    # grid_search ,acc , best_rf =runclass( model=model,param_grid=param_grid ,data=data)

    # # X_train, X_test, y_train, y_test = dataprocess(path=path,metircs=metircs,usearea=usearea,usenorm=usenorm)
    # # data=[X_train, X_test, y_train, y_test ]
    #     #保存数据  
    # # print('saving data ...')

    # # savedatapath  = f'/home/dls/data/openmmlab/test_RandomForestRegressor/{basename}_{timestamp}_data'
    # # print(f'datapath {datapath}')
    # # savedata(data=data,datapath=savedatapath)
    # # 进行因变量分箱
   
    # # # 保存grid search
    # model_name = basename+f'_{round(acc,3)}_{timestamp}.pkl'
    # gs_name = basename+f'_gs_{round(acc,3)}_{timestamp}.pkl'

    # os.makedirs(savemodelpath,exist_ok=True)
    # model_name = os.path.join(savemodelpath, model_name)
    # gs_name = os.path.join(savemodelpath, gs_name)
    # print(f'save model_name {model_name}')
    # print(f'save gs_name {gs_name}')

    # save_grid_search(best_rf,model_name)
    # save_grid_search(grid_search,gs_name)

    # # 训练随机森林分类模型
   
    # model.fit(X_train, y_train)



    # y_pred = best_rf.predict(X_test)
    
    # # 绘制训练结果和测试结果的图表
    # plt.scatter(y_train, best_rf.predict(X_train), label='Train')
    # plt.scatter(y_test, y_pred, label='Test')
    # plt.plot([0.5, 1], [0.5, 1], '--k', label='Ideal')
    # plt.xlabel('True Values')
    # plt.ylabel('Predictions')
    # plt.legend()
    # picpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/pic'
    # plt.savefig(os.path.join(picpath,f'{basename}.png'))
def mainxBoostingClassifier2( datapath,tag ,savemodelpath ):
    # param_grid = {
    #     'learning_rate': [  0.1, 0.01,0.001,0.0001],
    #     'n_estimators': [    1000,1500,2000 ],
    #     'max_depth': [  3,6,10,100],
    #     'min_samples_split':[  0.0001, 0.01,0.5 ],
    #     'min_samples_leaf':[    0.0001, 0.01,0.5 ],
    #     'warm_start':[True ],
    #     'max_features' : [   'sqrt', 'log2',None  ],
    #     # 'min_samples_leaf':[20],
    #     'subsample':[1.0 ,0.8 ],
    #     # 'n_iter_no_change': [5, 10, None]
    # }
  
    # param_grid = {
    #     'learning_rate': [ 0.001,  0.01, 0.1],
    #     'max_depth': [3, 7,10,None ],
    #     # 'booster':['gbtree','gblinear'],
    #     'n_estimators': [ 100,1000,5000 ],
    #     'subsample': [ 0.6, 0.8, 1.0],
    #     'colsample_bytree': [ 0.6, 0.8, 1.0] ,
    #     'gamma': [0,     0.5, 1 ],
    #     'reg_alpha': [0, 1e-4, 1e-2],
    #     'reg_lambda': [0, 1e-4, 1e-2],
    #     # 'gpu_id ':[0]
    # }

    param_grid = {
        'learning_rate': [   0.01],
        'max_depth': [20 ],
        # 'booster':['gbtree','gblinear'],
        'n_estimators': [ 1000  ],
        # 'subsample': [  0.5],
        # 'min_samples_split':[  0.0001, 0.01,0.5 ],
        # 'min_samples_leaf':[    0.0001, 0.01,0.5 ],
        # 'colsample_bytree': [   0.5] ,
        # 'gamma': [  1 ],
        # 'reg_alpha': [ 1e-1],
        # 'reg_lambda': [ 1e-1],
        # 'gpu_id ':[0]
    }
    basename = f'XGBC_{tag}' 
    print(f'basename {basename}')
    timestamp =  time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp ))

    # data = pd.read_csv(datapath)
    # 对数据进行处理  根据相关性 筛选特征 并进行归一化
    data = dataprocessdata_rdp(datapath)
    
    namex = data.columns[:-1]
    namex = [name  for name in namex if '_maxmin' not in name]
    namey = data.columns[-1]
    print('namex  ',namex )
    print('namey', namey)
    # exit()

    datax = data[namex]
    datay = data[namey]
    nbins =10
    kbins = KBinsDiscretizer(n_bins=nbins, strategy='uniform',encode='ordinal')
    # kbins = KBinsDiscretizer(n_bins=10, strategy='uniform',encode='onehot')
    y_binned = kbins.fit_transform(datay.to_numpy() .reshape(-1, 1) ) .squeeze()
    sns.histplot(datay, kde=True,bins=nbins )
    plt.savefig( f'{basename}_y_binned{nbins}.png')
    # print('y_binned',y_binned.shape )   
    # datax_filter =datax_filter.loc[:,~datax_filter.columns.str.contains('_maxmin')]
    # datax_filter =datax_filter.loc[:,~datax_filter.columns.str.contains('_std')]
    # datax_filter =datax_filter.loc[:,~datax_filter.columns.str.contains('per')]
    # print('namey',namey)
    # exit()
    # weight = compute_sample_weight('balanced', y_binned)
    X_train, X_test, y_train, y_test = train_test_split(datax,y_binned, test_size=0.2, random_state=randomseed)
    data=[X_train, X_test, y_train, y_test ]
    # weight = compute_sample_weight('balanced', y_train)
    # weight = compute_class_weight(class_weight='balanced', classes= np.unique(y_train) , y=y_train.squeeze() )
    # results = []
    # for params in tqdm(ParameterGrid(param_grid)):
    #     result={}
    #     model = GradientBoostingClassifier(**params)
    #     # exit()
    #     model.fit(X_train,y_train, sample_weight=weight)
    #     # 对新数据进行预测
    #     y_pred_binned = model.predict(X_test)
    #     # print('y_pred_binned',y_pred_binned.shape )
    #     # print('y_test',y_test.shape )
    #     # accuracy =model.score(y_pred_binned.reshape(-1, 1), y_test.reshape(-1, 1))
    #     accuracy = accuracy_score(y_test, y_pred_binned)
    #     result['acc'] = accuracy
    #     for key ,value in params.items():
    #         result[key] =value
    #     results.append(result)
    # 定义函数
    def fit_model(params):
        result = {}
        model = xgb.XGBClassifier(**params,random_state=randomseed)
        # model.fit(X_train, y_train, sample_weight=weight)
        model.fit(X_train, y_train, sample_weight=weight)

        y_pred_binned = model.predict(X_test)


        accuracy = accuracy_score(y_test, y_pred_binned)
        result['acc'] = accuracy
        for key, value in params.items():
            result[key] = value
        # print(result)
        return result
    def parallelrun():
        # 创建参数列表
        param_list = list(ParameterGrid(param_grid))

        # 使用 joblib 并行执行函数
        results = Parallel(n_jobs=16, verbose=0)(delayed(fit_model)(params) for params in tqdm(param_list))
        # results = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(params) for params in param_list)
# 

        df= pd.DataFrame(results)
        df.to_csv(f'{basename}_{timestamp}_XGBCresults2.csv',index=False)
        print(df)
        
        # 获取 age 列最大值所在的行
        max_age_index = df['acc'].idxmax()
        max_age_row = df.loc[max_age_index]

        # 打印结果
        print(max_age_row)
    def testrun():
        results = []
        # for params in tqdm(ParameterGrid(param_grid)):
        for params in  ParameterGrid(param_grid) :

            result={}
            model =  xgb.XGBClassifier(**params,objective='multi:softmax', num_class=nbins)
            # exit()
            # model.fit(X_train,y_train, sample_weight=weight)
            model.fit(X_train,y_train )

            # 对新数据进行预测
            y_pred_binned = model.predict(X_test)
            # print('y_pred_binned',y_pred_binned )
            # print('y_test',y_test.shape )
            # accuracy =model.score(y_pred_binned.reshape(-1, 1), y_test.reshape(-1, 1))
            accuracy = accuracy_score(y_test, y_pred_binned)
            result['acc'] = accuracy
            y_train_binned = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_binned)
            result['train_acc'] = train_accuracy
            for key ,value in params.items():
                result[key] =value
            print(result)
            results.append(result)

        df= pd.DataFrame(results)
        # df.to_csv(f'{basename}_{timestamp}_gbcresults.csv',index=False)
        print(df)
        # 获取 age 列最大值所在的行
        max_age_index = df['acc'].idxmax()
        max_age_row = df.loc[max_age_index]

        # 打印结果
        print(max_age_row)
    # parallelrun()
    testrun()
    # grid_search ,acc , best_rf =runclass( model=model,param_grid=param_grid ,data=data)

    # # X_train, X_test, y_train, y_test = dataprocess(path=path,metircs=metircs,usearea=usearea,usenorm=usenorm)
    # # data=[X_train, X_test, y_train, y_test ]
    #     #保存数据  
    # # print('saving data ...')

    # # savedatapath  = f'/home/dls/data/openmmlab/test_RandomForestRegressor/{basename}_{timestamp}_data'
    # # print(f'datapath {datapath}')
    # # savedata(data=data,datapath=savedatapath)
    # # 进行因变量分箱
   
    # # # 保存grid search
    # model_name = basename+f'_{round(acc,3)}_{timestamp}.pkl'
    # gs_name = basename+f'_gs_{round(acc,3)}_{timestamp}.pkl'

    # os.makedirs(savemodelpath,exist_ok=True)
    # model_name = os.path.join(savemodelpath, model_name)
    # gs_name = os.path.join(savemodelpath, gs_name)
    # print(f'save model_name {model_name}')
    # print(f'save gs_name {gs_name}')

    # save_grid_search(best_rf,model_name)
    # save_grid_search(grid_search,gs_name)

    # # 训练随机森林分类模型
   
    # model.fit(X_train, y_train)



    # y_pred = best_rf.predict(X_test)
    
    # # 绘制训练结果和测试结果的图表
    # plt.scatter(y_train, best_rf.predict(X_train), label='Train')
    # plt.scatter(y_test, y_pred, label='Test')
    # plt.plot([0.5, 1], [0.5, 1], '--k', label='Ideal')
    # plt.xlabel('True Values')
    # plt.ylabel('Predictions')
    # plt.legend()
    # picpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/pic'
    # plt.savefig(os.path.join(picpath,f'{basename}.png'))
 
def main():
  # mainrf()
    # maingbt()
    
    # for usearea in [False,True]:
    #     maingbt(usearea)
    #     mainrf(usearea)
    # for usearea in [False,True]:
    #     mainabr(usearea)
    path = '/home/dls/data/openmmlab/test_RandomForestRegressor/merge_all.csv'#小数据集
    # path = '/home/dls/data/openmmlab/test_RandomForestRegressor/large_samples_b60_all.csv'
    tag = ''
    metircs='f1s_norm'
    usenorm = False
    for usearea in [ False]:
        mainxboost(usearea,path,tag=tag,metircs=metircs,usenorm=usenorm)
def main1():
  # mainrf()
    # maingbt()
    
    # for usearea in [False,True]:
    #     maingbt(usearea)
    #     mainrf(usearea)
    # for usearea in [False,True]:
    #     mainabr(usearea)
    # datapath = '/home/dls/data/openmmlab/test_RandomForestRegressor/calrel/calrelout.csv' 
    datapath = '/home/dls/data/openmmlab/test_RandomForestRegressor/calrel/calrelout_withstd.csv'
    # path = '/home/dls/data/openmmlab/test_RandomForestRegressor/large_samples_b60_all.csv'
    tag = ''
    savemodelpath ='/home/dls/data/openmmlab/test_RandomForestRegressor/bestmodel'
    for usearea in [ True]:
        mainGradientBoostingRegressor(usearea,datapath,tag=tag,savemodelpath=savemodelpath)
        # mainxboost1(usearea,datapath,tag=tag,savemodelpath=savemodelpath )
def mainclasseir():
  # mainrf()
    # maingbt()
    
    # for usearea in [False,True]:
    #     maingbt(usearea)
    #     mainrf(usearea)
    # for usearea in [False,True]:
    #     mainabr(usearea)
    # datapath = '/home/dls/data/openmmlab/test_RandomForestRegressor/calrel/calrelout.csv' 
    datapath = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_rdp_l5/dmdoutbygrid/FULLER4D_5_cnngeo.csv'
    # path = '/home/dls/data/openmmlab/test_RandomForestRegressor/large_samples_b60_all.csv'
    tag = ''
    savemodelpath ='/home/dls/data/openmmlab/test_RandomForestRegressor/bestmodel'
 
        # mainGradientBoostingRegressor(usearea,datapath,tag=tag,savemodelpath=savemodelpath)
    mainGradientBoostingClassifier( datapath,tag=tag,savemodelpath=savemodelpath)
    # mainxBoostingClassifier( datapath,tag=tag,savemodelpath=savemodelpath)
        # mainxboost1(usearea,datapath,tag=tag,savemodelpath=savemodelpath )
def maindataprocess():
    processname = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_rdp_l5/dmdoutbygrid/FULLER4D_5_cnngeo.csv'
    # dataprocessdata_rdp(processname=processname)
    # datapeocdessdata_rdp2(processname=processname)
    dataprocessdata_rdp_normx(processname=processname)
# 分类和回归一起做
def mainxboostnew ():
    datapath = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_rdp_l5/dmdoutbygrid/FULLER4D_5_cnngeo.csv'
    # path = '/home/dls/data/openmmlab/test_RandomForestRegressor/large_samples_b60_all.csv'
    tag = ''
    savemodelpath ='/home/dls/data/openmmlab/test_RandomForestRegressor/bestmodel'
    mainxboost2( datapath,tag=tag,savemodelpath=savemodelpath)
    # mainxBoostingClassifier( datapath,tag=tag,savemodelpath=savemodelpath)
    # mainxBoostingClassifier2(datapath,tag=tag,savemodelpath=savemodelpath)

def maingbr():
    datapath = '/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_rdp_l5/dmdoutbygrid/FULLER4D_5_cnngeo.csv'
    # path = '/home/dls/data/openmmlab/test_RandomForestRegressor/large_samples_b60_all.csv'
    tag = ''
    savemodelpath ='/home/dls/data/openmmlab/test_RandomForestRegressor/bestmodel'
    # mainGradientBoostingRegressorpca(datapath,tag=tag,savemodelpath=savemodelpath)
    # mainGradientBoostingRegressor2( datapath,tag=tag,savemodelpath=savemodelpath)
    # mainGradientBoostingRegressor3( datapath,tag=tag,savemodelpath=savemodelpath)
    mainGradientBoostingRegressor4( datapath,tag=tag,savemodelpath=savemodelpath)


if __name__ == '__main__':
    # mainclasseir()
    # mainxboostnew()
    maingbr()
    # maindataprocess()
    # plotbins('/home/dls/data/openmmlab/mmclassification/tools/pandas_save/r18_ucm_f1_fuller4dmd_rdp_l5/dmdoutbygrid/FULLER4D_5_cnngeo.csv')
    # tescalrel()
    # tescalrel1()

  