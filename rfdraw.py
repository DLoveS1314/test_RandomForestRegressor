# 随机森林出图
from save_model import get_grid_search
import os
import pandas as pd
from sklearn.metrics import r2_score 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def modelacctest():
    # 测试一下 模型是否精度是否正确
    modelname = '/home/dls/data/openmmlab/test_RandomForestRegressor/bestmodel/GBR_4__0.795_20230511_132404.pkl'
    dataname = '/home/dls/data/openmmlab/test_RandomForestRegressor/traindata/GBR_4__20230511_132404_0.7945490403851381_data'
    model= get_grid_search(modelname)
    y_testpath  =os.path.join(dataname,'y_test.csv')
    x_testpath  =os.path.join(dataname,'X_test.csv')

    y_test = pd.read_csv(y_testpath )
    x_test = pd.read_csv(x_testpath )

    y_pred = model.predict(x_test)
    accuracy = r2_score(y_test, y_pred)
    print('accuracy:', accuracy)
def getmodel_data(path,datapath):
    model = get_grid_search(path)
    y_testpath  =os.path.join(datapath,'y_test.csv')
    x_testpath  =os.path.join(datapath,'X_test.csv')

    y_test = pd.read_csv(y_testpath )
    x_test = pd.read_csv(x_testpath )
    return model,y_test,x_test
def get_traindata(datapath):
    y_trainpath  =os.path.join(datapath,'y_train.csv')
    x_trainpath  =os.path.join(datapath,'X_train.csv')
    y_train = pd.read_csv(y_trainpath )
    x_train = pd.read_csv(x_trainpath )
    return x_train,y_train
def drawfeature_importances(model,columns):
    # 获取特征重要性得分
    importances = model.feature_importances_

    # 绘制条形图
    plt.title("Feature importance")
    indices = np.argsort(importances)[::-1] # 降序排列
    x = list(range(len(importances)))
    x_label = columns[indices]
    plt.figure(figsize=(8,8))
    y  = importances[indices]
    color= np.array([73, 151, 201,255.0 ])/255.0
    sns.barplot(x=y, y=x ,orient='h' ,color=color,errorbar="sd")
    # sns.scatterplot(x=y, y=x   ,color=color)

    plt.yticks(x, x_label)
    plt.tight_layout()
    plt.savefig('./rfdraw/feature_importances.png',  transparent=True)
    # plt.savefig('./rfdraw/feature_importances.png' )
from sklearn.inspection import plot_partial_dependence,PartialDependenceDisplay
def ppd(model,x_data):
    # plot_partial_dependence 部分变量依赖图
    plt.figure(figsize=(8,8))
   
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(9, 10))
    sns.color_palette("Paired")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="white", rc=custom_params)
    # 循环绘制子图
    for i, ax in enumerate(axes.flatten()):
        # 如果是最后一行，则只绘制一列        ax.tick_params(labelsize=18) #刻度字体大小13
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(16)
        # mean =data[cols[i]].mean()
        # std = data[cols[i]].std()
        # min = data[cols[i]].min()
        # max = data[cols[i]].max()
        # values = np.linspace(min,max,4)
        if i %3 ==0:
            ax.set_ylabel('accs',fontsize=18)
        else:
            ax.set_ylabel('')
        # 设置colorbar的属性
        # ticks = [mean-2*std,mean-std,mean,mean+std,mean+2*std]
        ax.set_title(' ',pad=2)
        axout=PartialDependenceDisplay.from_estimator(model, x_data,features = [i],ax=ax)  

        # ax.set_xticks([round(value,4) if value<1.0 else round(value,2) for value in values ])
    # 调整子图间距
    # fig.tight_layout(rect=(0.01, 0, 1, 1) )
    fig.tight_layout(  )

    path ='/home/dls/data/openmmlab/test_RandomForestRegressor/rfdraw/partial_dependence.png'
    # 显示图形
    plt.savefig(path)
from scipy.stats import linregress
def scatterplot(model,x_test,x_train,y_train,y_test):
    # 绘制训练结果和测试结果的图表
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    print(y_test.shape)
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    # sns.relplot(  x=y_train, y=model.predict(x_train) )
    color= np.array([73, 151, 201,255.0 ])/255.0
    colort= np.array([221, 132, 82,255.0 ])/255.0 





    sns.regplot(  x=y_test.to_numpy().flatten() , y=y_pred.flatten(),scatter_kws={"s": 40   },ax=ax ,label='test',color=color)
    sns.regplot(  x=y_train.to_numpy().flatten(), y=y_pred_train.flatten(),scatter_kws={"s": 40, 'alpha':.4 },ax=ax,label='train',color=colort)  


    slope, intercept, r_value, p_value, std_err = linregress(y_test.to_numpy().flatten(), y_pred.flatten())
    teststr ="y_test = {:.2f}X  ".format(slope )
    slopet, interceptt, r_value, p_value, std_err = linregress(y_train.to_numpy().flatten(), y_pred_train.flatten())
    trainstr="y_train = {:.2f}X  ".format(slopet )
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.text( ax.get_xlim()[0] +x_range*0.13 ,ax.get_ylim()[0] +y_range*0.9 ,   trainstr+r' $R^2$ =0.82', ha='center', va='center',fontsize =10    )

    ax.text( ax.get_xlim()[0] +x_range*0.13 ,ax.get_ylim()[0] +y_range*0.86,   teststr+r'  $R^2$ =0.80', ha='center', va='center',fontsize =10 )# ,fontweight='heavy' 
    # ax.axline(xy1=(0, 0), slope=1, color="b" )
    ax.axline(xy1=( ax.get_xlim()[0],ax.get_ylim()[0]), xy2=(ax.get_xlim()[1],ax.get_ylim()[1]), color='k', linestyle='--', linewidth=1) 

    ax.set_xticks(np.arange (87,90.5,0.5))
    ax.set_yticks(np.arange(87,90.5,0.5))

   
    # plt.scatter(y_train, model.predict(x_train), label='Train')
    # plt.scatter(y_test, model.predict(x_test), label='Test')
    # plt.plot([0.5, 1], [0.5, 1], '--k', label='Ideal')
    plt.legend()
 
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.savefig('./rfdraw/relplot.png')
def main():
    modelname = '/home/dls/data/openmmlab/test_RandomForestRegressor/bestmodel/GBR_4__0.795_20230511_132404.pkl'
    datapath = '/home/dls/data/openmmlab/test_RandomForestRegressor/traindata/GBR_4__20230511_132404_0.7945490403851381_data'
    x_train,y_train =get_traindata(datapath=datapath)
    model,y_test,x_test =getmodel_data(modelname,datapath=datapath)
    x_data= pd.concat([x_train,x_test],axis=0)
    print(x_data.columns)
    print(x_data.shape)

    # 绘制特征重要性
    # columns =x_test.columns
    # drawfeature_importances(model,columns)
    # 绘制部分变量依赖图
    # ppd(model=model,x_data=x_data)

    # 绘制测试结果散点图
    scatterplot(model,x_test,x_train,y_train,y_test)
if __name__ == '__main__':
 
    main()