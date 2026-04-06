import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys
import os
import seaborn as sns
import logging
from logging_code import setup_logging
logger = setup_logging('hyperparameter_tuning')


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV,cross_validate

def tuning(Training_data_bal_sc,y_train_bal, Testing_data_Zscore, y_test):
    try:
        parameters_list = [
            # ----- L2 -----
            {
                'penalty': ['l2'],
                'solver': ['sag', 'saga', 'liblinear'],
                'C': [0.1, 1, 10, 100],
                'max_iter': [100, 200, 500],
                'class_weight': [None, 'balanced'],
            },

            # ----- L1 -----
            {
                'penalty': ['l1'],
                'solver': ['liblinear', 'saga'],
                'C': [0.01, 0.1, 1, 10, 100],
                'max_iter': [100, 200, 500],
                'class_weight': [None, 'balanced'],
            }
        ]

        grid_reg = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters_list, scoring='f1', cv=5)
        grid_result = grid_reg.fit(Training_data_bal_sc, y_train_bal)
        logger.info(f'The grid_result {grid_result}')
        logger.info(f'The grid best parameter are {grid_result.best_params_}')
        logger.info(f'The grid Accuracy Score {grid_result.best_score_}')


    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')


# best parameters are this
#{'C': 100, 'class_weight': 'balanced', 'max_iter': 500, 'penalty': 'l1', 'solver': 'liblinear'}