import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sklearn
import sys
import seaborn as sns
import warnings

from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
from logging_code import setup_logging
logger=setup_logging('feature_scaling')
import sklearn
import sys
from sklearn.preprocessing import StandardScaler
from all_models import common
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle
def fs(X_train,y_train,X_test,y_test):
    try:
        logger.info(f"Training data  independent : {X_train.shape}")
        logger.info(f"Training data  dependent : {y_train.shape}")
        logger.info(f"Testing data  independent : {X_test.shape}")
        logger.info(f"Testing data  dependent : {y_test.shape}")
        logger.info(f"before : {X_train.head(1)}")


        sc=StandardScaler()
        sc.fit(X_train)
        X_train_sc = sc.transform(X_train)
        X_test_sc = sc.transform(X_test)

        with open('standscaler.pkl','wb') as f:
            pickle.dump(sc, f)


        logger.info(f"before : {X_train_sc}")


        common(X_train_sc,y_train,X_test_sc,y_test)
        reg = LogisticRegression(C= 100, class_weight= 'balanced', max_iter= 500, penalty= 'l1', solver= 'liblinear')
        reg.fit(X_train_sc, y_train)


        logger.info(f"test accuracy : {accuracy_score(y_test, reg.predict(X_test_sc))}")
        logger.info(f"test confusion matrix : {confusion_matrix(y_test, reg.predict(X_test_sc))}")
        logger.info(f"classification report : {classification_report(y_test, reg.predict(X_test_sc))}")

        with open('MODEL.pkl','wb') as f:
            pickle.dump(reg, f)

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
