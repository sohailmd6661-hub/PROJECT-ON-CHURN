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
logger = setup_logging('cat_to_num')

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def c_t_n(X_train_cat, X_test_cat):
    try:
        # ================= NOMINAL =================
        logger.info(f'before nominal X_train_cat:{X_train_cat.shape} \n{X_train_cat.columns}')
        logger.info(f'before nominal X_test_cat:{X_test_cat.shape} \n{X_test_cat.columns}')


        oh = OneHotEncoder(drop='first')

        oh.fit(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'Sim']])

        values_train = oh.transform(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'Sim']]).toarray()
        values_test = oh.transform(X_test_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'Sim']]).toarray()

        t_train = pd.DataFrame(values_train,columns=oh.get_feature_names_out())
        t_test = pd.DataFrame(values_test,columns=oh.get_feature_names_out())


        X_train_cat.reset_index(drop=True, inplace=True)
        X_test_cat.reset_index(drop=True, inplace=True)

        t_train.reset_index(drop=True, inplace=True)
        t_test.reset_index(drop=True, inplace=True)

        X_train_cat = pd.concat([X_train_cat, t_train], axis=1)
        X_test_cat = pd.concat([X_test_cat, t_test], axis=1)

        drop_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'Sim']

        X_train_cat = X_train_cat.drop(drop_cols, axis=1)
        X_test_cat = X_test_cat.drop(drop_cols, axis=1)

        logger.info(f'after nominal X_train_cat:{X_train_cat.shape}\n{X_train_cat.columns}')
        logger.info(f'after nominal X_test_cat:{X_test_cat.shape}\n{X_test_cat.columns}')

        # ================= ORDINAL =================
        logger.info("==================== ordinal ====================")

        od = OrdinalEncoder()

        od.fit(X_train_cat[['Contract']])

        ord_train = od.transform(X_train_cat[['Contract']])
        ord_test = od.transform(X_test_cat[['Contract']])

        t1o_train = pd.DataFrame(ord_train, columns=['Contract_re'])
        t2o_test = pd.DataFrame(ord_test, columns=['Contract_re'])

        X_train_cat.reset_index(drop=True, inplace=True)
        X_test_cat.reset_index(drop=True, inplace=True)

        t1o_train.reset_index(drop=True, inplace=True)
        t2o_test.reset_index(drop=True, inplace=True)

        X_train_cat = pd.concat([X_train_cat, t1o_train], axis=1)
        X_test_cat = pd.concat([X_test_cat, t2o_test], axis=1)

        X_train_cat = X_train_cat.drop(['Contract'], axis=1)
        X_test_cat = X_test_cat.drop(['Contract'], axis=1)

        logger.info(f'after ordinal X_train_cat:{X_train_cat.shape}\n{X_train_cat.columns}')
        logger.info(f'after ordinal X_test_cat:{X_test_cat.shape}\n{X_test_cat.columns}')

        logger.info(f'null values in X_train_cat:\n{X_train_cat.isnull().sum()}')
        logger.info(f'null values in X_test_cat:\n{X_test_cat.isnull().sum()}')


        return X_train_cat, X_test_cat


    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error: {e}")
        logger.error(f"Error in line no: {er_line.tb_lineno}")