import pandas as pd
from factor_analyzer import FactorAnalyzer
# 计算巴特利特P值
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo


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