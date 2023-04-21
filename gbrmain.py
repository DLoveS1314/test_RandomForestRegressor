from sklearn.ensemble import RandomForestRegressor  ,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from save_model import *

# 梯度提升树
# GradientBoostingRegressor是一种基于决策树的梯度提升算法，可以用于回归问题。下面是一些可能需要调整的重要参数：

# n_estimators：弱学习器的数量（即决策树的数量）。默认值为100，增加该值通常可以提高模型的准确性，但也会增加计算成本。

# learning_rate：每个决策树的权重缩减系数。默认值为0.1，通常应该在0.01至0.2之间。较小的学习率会使模型收敛速度变慢，但可以提高泛化能力；较大的学习率会使模型更快地收敛，但可能导致过拟合。

# max_depth：决策树的最大深度。默认值为3，通常应该在2至8之间。增加最大深度可以提高模型的准确性，但也可能导致过拟合。

# min_samples_split ：决策树的最小样本分割数量。默认值为2，通常应该在100至1000之间。增加该值可以减少过拟合。

# min_samples_leaf：决策树的最小叶节点样本数量。默认值为1，通常应该在10至100之间。增加该值可以减少过拟合。

# subsample ： 每个决策树的样本采样比例。默认值为1，表示使用所有样本进行训练。减小该值可以减少过拟合。


# 加载数据集
# from sklearn.ensemble import RandomForestRegressor  ,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# 加载数据集
path ='/home/dls/data/openmmlab/test_RandomForestRegressor/merge.csv'
data = pd.read_csv(path)

data_x =  data.iloc[:, :-1]
data_y =  data.iloc[:,  -1]
# print(data_x.head())
# print(data_y.head())

# need_name = ['area','per','zsc','disminmax','disavg','angleminmax','angleavg','csdminmax','csdavg','precisions' ]
# data =data[need_name]
# 将数据集分成训练集和测试集 最后一列是因变量 剩余的列是自变量
scaler = MinMaxScaler() #为了使用同一个归一化器 先归一化 再分割
df_normalized = pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns)
X_train, X_test, y_train, y_test = train_test_split(df_normalized, data_y, test_size=0.2, random_state=42)

# 定义参数空间 把早停放进去
param_grid = {
    'learning_rate': [0.05, 0.01,0.02],
    'n_estimators': [ 500,1000,2000,5000],
    'max_depth': [3, 4, 5,6,7,8],
    'min_samples_split':[ 10,50,100],
    'random_state':[ 19960229]
    # 'n_iter_no_change': [5, 10, None]
}
# 早停
# model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, 
#                                   subsample=0.8, max_depth=3, validation_fraction=0.2, n_iter_no_change=5, tol=0.0001, random_state=42, verbose=1)
# 初始化梯度提升树回归器
gbt = GradientBoostingRegressor()

 
# 网格搜索
grid_search = GridSearchCV(gbt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2,return_train_score=True)
grid_search.fit(X_train, y_train)

# 输出最佳参数组合
print('Best parameters: ', grid_search.best_params_)
save_grid_search(grid_search)

# 用最佳参数组合训练模型
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
# 用最佳参数组合对测试集进行预测
y_pred = best_rf.predict(X_test)

# 计算R2和RMSE
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# 输出R2和RMSE
print('R2 score: {:.2f}'.format(r2))
print('RMSE: {:.2f}'.format(rmse))

# 保存权重
import joblib
joblib.dump(gbt, 'rf_regressor.pkl')

# 绘制训练结果和测试结果的图表
plt.scatter(y_train, best_rf.predict(X_train), label='Train')
plt.scatter(y_test, y_pred, label='Test')
plt.plot([0.5, 1], [0.5, 1], '--k', label='Ideal')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.savefig('figure.png')

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