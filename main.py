from sklearn.ensemble import RandomForestRegressor  ,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from save_model import *
import time
import os
def dataprocess(path,metircs,usearea=False,usenorm=False):
    # path ='/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/FULLER4D_6_process.csv'
    # path ='/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/FULLER4H_6_process.csv'
    # path ='/home/dls/data/openmmlab/test_RandomForestRegressor/processdata/FULLER4T_6_process.csv'
    # path = '/home/dls/data/openmmlab/test_RandomForestRegressor/merge_nopre_test.csv'
    # pd.set_option('display.max_columns', 1000)
    # pd.set_option('display.max_rows', 1000)
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
    print('x',data_x.columns)
    print('y',data_y .name)
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
    X_train.to_csv('./data/X_train.csv',index=False)
    X_test.to_csv('./data/X_test.csv',index=False)
    y_train.to_csv('./data/y_train.csv',index=False)
    y_test.to_csv('./data/y_test.csv',index=False)
    return X_train, X_test, y_train, y_test
# 加载数据集
def run(path,model,param_grid,metircs='f1s_norm',usearea=False,namefile='',usenorm=False):
    # 梯度提升树
    # GradientBoostingRegressor是一种基于决策树的梯度提升算法，可以用于回归问题。下面是一些可能需要调整的重要参数：

    # n_estimators：弱学习器的数量（即决策树的数量）。默认值为100，增加该值通常可以提高模型的准确性，但也会增加计算成本。

    # learning_rate：每个决策树的权重缩减系数。默认值为0.1，通常应该在0.01至0.2之间。较小的学习率会使模型收敛速度变慢，但可以提高泛化能力；较大的学习率会使模型更快地收敛，但可能导致过拟合。

    # max_depth：决策树的最大深度。默认值为3，通常应该在2至8之间。增加最大深度可以提高模型的准确性，但也可能导致过拟合。

    # min_samples_split ：决策树的最小样本分割数量。默认值为2，通常应该在100至1000之间。增加该值可以减少过拟合。

    # min_samples_leaf：决策树的最小叶节点样本数量。默认值为1，通常应该在10至100之间。增加该值可以减少过拟合。

    # subsample ： 每个决策树的样本采样比例。默认值为1，表示使用所有样本进行训练。减小该值可以减少过拟合。

   

    # 定义参数空间
    # param_grid = {
    #     'n_estimators': [10,50,100,500,1000],
    #     'max_depth': [    None],
    #     'min_samples_split': [1,2 ,4 ,10 ],
    #     'min_samples_leaf': [1, 2,4  ,10  ],
    #     'max_features': ['sqrt', 'log2', None],
    # }
    # 早停
    # model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, 
    #                                   subsample=0.8, max_depth=3, validation_fraction=0.2, n_iter_no_change=5, tol=0.0001, random_state=42, verbose=1)
    # 初始化梯度提升树回归器
    # mdoel =  RandomForestRegressor(random_state=19960229)
    X_train, X_test, y_train, y_test = dataprocess(path=path,metircs=metircs,usearea=usearea,usenorm=usenorm)
    
    # 网格搜索
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0,return_train_score=True)
    grid_search.fit(X_train, y_train)
# XGBoost  结合 sklearn使用 GridSearchCV
    # 输出最佳参数组合
    # name = 'rf_grid_sea.pkl'
    timestamp =  time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp ))
    if usearea:
        basename = f'{model.__class__}_usearea_{namefile}'.split('.')[-1]
    else:
        basename = f'{model.__class__}_noarea_{namefile}'.split('.')[-1]


    # 用最佳参数组合训练模型
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train, y_train)
    # 用最佳参数组合对测试集进行预测
    y_pred = best_rf.predict(X_test)

    # 计算R2和RMSE
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    search_name = basename+f'{timestamp}_{round(r2,3)}.pkl'
    csv_name = basename+f'{timestamp}_{round(r2,3)}.csv'

  
    modelpath = '/home/dls/data/openmmlab/test_RandomForestRegressor/bestmodel'
    namepath = os.path.join(modelpath, search_name)
    save_grid_search(grid_search,namepath)
    df =pd.DataFrame.from_dict(grid_search.cv_results_)

    csvpath = os.path.join(modelpath,csv_name)
    df.to_csv(csvpath)
    logname = f'{basename}_log_{timestamp}'
    with open(f'/home/dls/data/openmmlab/test_RandomForestRegressor/log/{logname}.txt','a+') as f :
        strval = f'Best parameters:  {grid_search.best_params_} \n'
        strval1 = 'R2 score: {:.2f}\n'.format(r2)
        strval2 = 'RMSE: {:.2f}\n'.format(rmse)
        f.write(strval)
        f.write(strval1)
        f.write(strval2)
        f.write(str(X_train.columns))
        f.write(str(y_train.name))
        f.write(f'use norm{usenorm} ')
        # 输出R2和RMSE
        print(strval)
        print('R2 score: {:.2f}'.format(r2))
        print('RMSE: {:.2f}'.format(rmse))

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

    # # 格网搜索加早停 
    # # 创建 GridSearchCV 对象，启用早停技术
    # grid_search = GridSearchCV(model, param_dist, cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error', return_train_score=True, refit=True, 
    #                              error_score='raise' )


    # # 'n_iter_no_change'被设置为一个正整数n时，如果在训练过程中，模型在连续n次迭代中都没有提升，则训练过程会提前结束；
    # # 定义网格搜索
    # grid_search = GridSearchCV(
    #     estimator=gbt,
    #     param_grid=param_grid,
    #     cv=5,
    #     scoring='neg_mean_squared_error',
    #     n_jobs=-1
    # )

    # # 执行网格搜索
    # grid_search.fit(X_train, y_train)

    
    # # 输出最佳参数和最佳模型的性能
    # print('Best params:', grid_search.best_params_)
    # print('Best score:', -grid_search.best_score_)
def maingbt(usearea):
    # path ='/home/dls/data/openmmlab/test_RandomForestRegressor/merge.csv'
    path = '/home/dls/data/openmmlab/test_RandomForestRegressor/merge_all.csv'
    gbt =  GradientBoostingRegressor(random_state=1997317)
    # 定义参数空间 把早停放进去
    param_grid = {
        'learning_rate': [0.05, 0.01,0.02],
        'n_estimators': [ 500,1000,2000,5000],

        'max_depth': [3, 4, 5,6,7,8],
        'min_samples_split':[ 10,50,100],
        # 'n_iter_no_change': [5, 10, None]
    }
    metircs='f1s_norm'
    run(path,model=gbt,param_grid=param_grid,metircs=metircs,usearea=usearea)
def mainrf(usearea):
    # path ='/home/dls/data/openmmlab/test_RandomForestRegressor/merge.csv'
    path = '/home/dls/data/openmmlab/test_RandomForestRegressor/merge_all.csv'
    rf =  RandomForestRegressor(random_state=1997317)
    param_grid = {
        'n_estimators': [500,1000,2000],
        'max_depth': [    None],
        'min_samples_split': [1,2 ,4 ,10 ],
        'min_samples_leaf': [1, 2,4  ,10  ],
        'max_features': ['sqrt', 'log2', None],
    }
    metircs='f1s_norm'
    run(path,model=rf,param_grid=param_grid,metircs=metircs,usearea=usearea)
def mainabr(usearea):
    from sklearn.tree import DecisionTreeRegressor 
    from sklearn.linear_model import LinearRegression
    # n_estimators：弱学习器的数量。
    # learning_rate：学习率，每个弱学习器的权重缩减系数。
    # loss：误差计算方法，可以选择linear、square或exponential。
    # base_estimator：弱学习器的基础模型，可以选择决策树模型或其他模型。
    # random_state：随机数种子，控制随机过程的复现性。
        # path ='/home/dls/data/openmmlab/test_RandomForestRegressor/merge.csv'
    path = '/home/dls/data/openmmlab/test_RandomForestRegressor/merge_all.csv'
    abr =  AdaBoostRegressor(random_state=1997317)
    param_grid = {'n_estimators': [   1000,3000,5000],
              'learning_rate': [0.02,0.01, 0.05],
              'loss': ['linear', 'square', 'exponential'],
              'estimator': [DecisionTreeRegressor(), LinearRegression() ]
            #   'estimator': [  RandomForestRegressor() ]
              }

    metircs='f1s_norm'
    run(path,model=abr,param_grid=param_grid,metircs=metircs,usearea=usearea)
def mainxboost(usearea,path):
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
    metircs='f1s_norm'
    namefile='largeb60'
    run(path,model=model,param_grid=param_grid,metircs=metircs,usearea=usearea,namefile=namefile)

if __name__ == '__main__':

    # mainrf()
    # maingbt()
    
    # for usearea in [False,True]:
    #     maingbt(usearea)
    #     mainrf(usearea)
    # for usearea in [False,True]:
    #     mainabr(usearea)
    # path = '/home/dls/data/openmmlab/test_RandomForestRegressor/merge_all.csv'
    path = '/home/dls/data/openmmlab/test_RandomForestRegressor/large_samples_b60_all.csv'
    for usearea in [True ,False]:
        mainxboost(usearea,path)