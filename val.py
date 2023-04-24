# 验证生成的模型 是否符合预期
from save_model import *
modepkl = '/home/dls/data/openmmlab/test_RandomForestRegressor/bestmodel/XGBR_na_un__0.956_20230423_105019.pkl'
searchmodel = get_grid_search(modepkl)