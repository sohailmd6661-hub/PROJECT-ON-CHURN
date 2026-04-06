import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sklearn
import sys
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from logging_code import setup_logging
logger=setup_logging('feature')
import sklearn
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

def feature_Select(x_train_num,x_test_num,y_train,y_test):
    try:
        logger.info(f'before train columns: {x_train_num.shape}\n {x_train_num.columns}')
        logger.info(f'before test columns: {x_test_num.shape}\n {x_test_num.columns}')
        reg = VarianceThreshold(threshold=0.01)
        reg.fit(x_train_num)
        logger.info(f'no of good columns in x_train_num.: {sum(reg.get_support())}:{x_train_num.columns[reg.get_support()]}')
        logger.info(f'no of bad columns x_train_num.: {sum(~reg.get_support())}:{x_train_num.columns[~reg.get_support()]}')

        logger.info(f'before train columns: {x_train_num.shape}\n {x_train_num.columns}')
        logger.info(f'before test columns: {x_test_num.shape}\n {x_test_num.columns}')
        logger.info(f'=============================== Hypothesis Testing =========================================')
        logger.info(f'=============================== Hypothesis Testing =========================================')



        c = []
        for i in x_train_num.columns:
            results = pearsonr(x_train_num[i], y_train)
            c.append(results)
        t = np.array(c)
        p_value = pd.Series(t[:, 1], index=x_train_num.columns)

        #f = []
        #p = 0
        #for val in p_value:
            #if val < 0.05:
                #f.append(x_train_num.columns[p])
            #p = p + 1
        #print(f)
        logger.info(f"After Train COlumns : {x_train_num.shape} \n : {x_train_num.columns}")
        logger.info(f"After Test COlumns : {x_test_num.shape} \n : {x_test_num.columns}")
        return x_train_num, x_test_num

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error: {e}")
        logger.error(f"Error in line no: {er_line.tb_lineno}")