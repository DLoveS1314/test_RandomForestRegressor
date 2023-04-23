from sklearn.ensemble import RandomForestRegressor  ,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from save_model import *
import time
import os
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
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=1997317)
    if usenorm :
        X_train, X_test, y_train, y_test = train_test_split(df_normalized, data_y, test_size=0.2, random_state=1997317)
    # X_train.to_csv(os.path.join(savedir,'X_train.csv') ,index=False)
    # X_test.to_csv(os.path.join(savedir,'X_test.csv') ,index=False)
    # y_train.to_csv(os.path.join(savedir,'y_train.csv') ,index=False)
    # y_test.to_csv(os.path.join(savedir,'y_test.csv') ,index=False)
    return X_train, X_test, y_train, y_test
# 加载数据集
def run( model,param_grid,data  ):
    X_train, X_test, y_train, y_test = data
    # 网格搜索
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0,return_train_score=True)
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

    return grid_search ,r2 ,rmse


    # # 保存权重
    # import joblib
    # joblib.dump(gbt, 'rf_regressor.pkl')

    # 绘制训练结果和测试结果的图表
    plt.scatter(y_train, best_rf.predict(X_train), label='Train')
    plt.scatter(y_test, y_pred, label='Test')
    plt.plot([0.5, 1], [0.5, 1], '--k', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.legend()
    picpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/pic'
    plt.savefig(os.path.join(picpath,f'{basename}.png'))

def savedata(data,datapath ):

    X_train, X_test, y_train, y_test =data 
    os.makedirs(datapath,exist_ok=True)
    X_train.to_csv(os.path.join(datapath,'X_train.csv') ,index=False)
    X_test.to_csv(os.path.join(datapath,'X_test.csv') ,index=False)
    y_train.to_csv(os.path.join(datapath,'y_train.csv') ,index=False)
    y_test.to_csv(os.path.join(datapath,'y_test.csv') ,index=False)



def mainxboost(usearea,path,tag='largeb60', metircs='f1s_norm',  usenorm = True):
    import xgboost as xgb
    # 设置 XGBoost 模型的参数空间
    # define XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=1997317
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
    grid_search ,r2 ,rmse = run(model=model,param_grid=param_grid,data=data )

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
    model_name = os.path.join(modelpath, model_name)
    print(f'model_name {model_name}')
    save_grid_search(grid_search,model_name)
if __name__ == '__main__':

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
    usenorm = True
    for usearea in [ False]:
        mainxboost(usearea,path,tag=tag,metircs=metircs,usenorm=usenorm)