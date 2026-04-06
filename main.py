from logging import info

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import sys
from sklearn.model_selection import train_test_split
from var_tran import vt_outliers
import logging
from logging_code import setup_logging
logger = setup_logging('main')   #creating a new logging file for main
from feature_scaling import fs
from feature import feature_Select
from handling_missing_values import handling_missing  #calling the missing values file
from cat_to_num import c_t_n
from imblearn.over_sampling import SMOTE




class CHURN:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path) #loading the dataset into df
            logger.info(self.df)
            logger.info(f'{self.df.info()}')
            logger.info(f'Before Updated dataset Size is: {self.df.shape}')

            # Adding a new column "sim" based on the Internet Service

            def add_sim(df):
                if df['PaymentMethod'] == 'Electronic check':
                    return 'Reliance Jio'
                elif df['PaymentMethod'] == 'Mailed check':
                    return 'Airtel'
                elif df['PaymentMethod'] == 'Bank transfer (automatic)':
                    return 'Vi-idea'
                else:
                    return 'BSNL'
            self.df['Sim'] = self.df.apply(add_sim,axis=1)

            logger.info(f'After updated file is {self.df}')
            logger.info(f'After updated dataset Size is: {self.df.shape}')
            logger.info(f'After updated dataset Size is: {self.df.columns}')

            logger.info(f'Checking for null values')
            for i in self.df.columns:
                logger.info(f'{i} -> {self.df[i].isnull().sum()}')


            logger.info(f'=================================================')

            # converting the total charge's into numeric
            self.df['TotalCharges'] = self.df['TotalCharges'].replace(" ", np.nan)
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'])
            logger.info(f'{self.df.info()}')

            for i in self.df.columns:
                logger.info(f'{i} -> {self.df[i].isnull().sum()}')


            #Divide the data into independent(X) and dependent(y)
            self.X = self.df.drop('Churn',axis=1)
            self.y = self.df['Churn']

            logger.info(f'checking the column names X : {self.X.columns}')
            logger.info(f'checking the shape of y : {self.y.shape}')

            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=45)

            #here converting the actual point from string(Yes/No) to numeric(1/0)
            self.y_train = self.y_train.map({'Yes':1,'No':0}).astype(int)
            self.y_test = self.y_test.map({'Yes': 1, 'No': 0}).astype(int)
            logger.info(f'Train data size {self.X_train.shape} and {self.y_train.shape}')
            logger.info(f'Test data size {self.X_test.shape} and {self.y_test.shape}')
            logger.info(self.y_train)
            logger.info(self.y_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def missing_values(self):
        try:
            logger.info(f'==================  Handling missing Values  =============================')
            logger.info(f'Before replacing Train null values : {self.X_train.isnull().sum()}')
            logger.info(f'Before replacing Test null values : {self.X_test.isnull().sum()}')

            self.X_train,self.X_test=handling_missing(self.X_train,self.X_test)

            logger.info(f'Before replacing Train null values : {self.X_train.isnull().sum()}')
            logger.info(f'Before replacing Test null values : {self.X_test.isnull().sum()}')


        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def data_separation(self):
        try:
            self.X_train_num_cols = self.X_train.select_dtypes(exclude='object')
            self.X_test_num_cols = self.X_test.select_dtypes(exclude='object')
            self.X_train_cat_cols = self.X_train.select_dtypes(include='object')
            self.X_test_cat_cols = self.X_test.select_dtypes(include='object')
            logger.info(f'X_train_num_cols:{self.X_train_num_cols.columns}:{self.X_train_num_cols.shape}')
            logger.info(f'X_test_num_cols:{self.X_test_num_cols.columns}:{self.X_test_num_cols.shape}')
            logger.info(f'X_train_cat_cols:{self.X_train_cat_cols.columns}:{self.X_train_cat_cols.shape}')
            logger.info(f'X_test_cat_cols:{self.X_test_cat_cols.columns}:{self.X_test_cat_cols.shape}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')




    def variable_transformation(self):

        try:
            logger.info(f'Before column names: {self.X_test_num_cols.columns}')
            self.X_train_num_cols, self.X_test_num_cols = vt_outliers(
            self.X_train_num_cols,
            self.X_test_num_cols
            )
            logger.info(f'After train column names: {self.X_train_num_cols.columns}')
            logger.info(f'After test column names: {self.X_test_num_cols.columns}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def feature_selection(self):
        try:
            self.X_train_num_cols, self.X_test_num_cols = feature_Select(self.X_train_num_cols,self.X_test_num_cols,self.y_train,self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error: {e}")
            logger.error(f"Error in line no: {er_line.tb_lineno}")

    def cat_to_num(self):
        try:
            self.X_train_cat_cols = self.X_train_cat_cols.drop(['customerID'],axis=1)
            self.X_test_cat_cols = self.X_test_cat_cols.drop(['customerID'], axis=1)
            self.X_train_cat_cols, self.X_test_cat_cols = c_t_n(self.X_train_cat_cols, self.X_test_cat_cols)
            self.X_train_num_cols.reset_index(drop=True, inplace=True)
            self.X_train_cat_cols.reset_index(drop=True, inplace=True)
            self.X_test_num_cols.reset_index(drop=True, inplace=True)
            self.X_test_cat_cols.reset_index(drop=True, inplace=True)
            self.training_data = pd.concat([self.X_train_num_cols, self.X_train_cat_cols], axis=1)
            self.testing_data = pd.concat([self.X_test_num_cols, self.X_test_cat_cols], axis=1)

            logger.info(f'final training data : {self.training_data.shape}')
            logger.info(f'nulls final training data : {self.training_data.isnull().sum()}')
            logger.info(f'final testing data : {self.testing_data.shape}')
            logger.info(f'nulls final testing data : {self.testing_data.isnull().sum()}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error: {e}")
            logger.error(f"Error in line no: {er_line.tb_lineno}")


    def data_balancing(self):
        try:
            logger.info(f"Before Number of Rows for Good Customer (1): {sum(self.y_train == 1)}")
            logger.info(f"Before Number of Rows for Bad Customer (0): {sum(self.y_train == 0)}")
            logger.info(f"Before balancing training_data: {self.training_data.shape}")
            self.training_data = self.training_data.apply(pd.to_numeric, errors='coerce')
            self.training_data.fillna(0, inplace=True)
            if self.training_data.select_dtypes(include=['object']).shape[1] > 0:

                raise Exception("Still contains non-numeric columns!")


            sm = SMOTE(random_state=42)
            self.training_data_bal, self.y_train_bal = sm.fit_resample(self.training_data, self.y_train)

            logger.info(f"After Number of Rows for Good Customer (1): {sum(self.y_train_bal == 1)}")
            logger.info(f"After Number of Rows for Bad Customer (0): {sum(self.y_train_bal == 0)}")
            logger.info(f"After balancing training_data: {self.training_data_bal.shape}")

            fs(self.training_data_bal, self.y_train_bal, self.testing_data, self.y_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error: {e}")
            logger.error(f"Error in line no: {er_line.tb_lineno}")




if __name__ == '__main__':
    try:
        obj = CHURN('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        obj.missing_values()
        obj.data_separation()
        obj.variable_transformation()
        obj.feature_selection()
        obj.cat_to_num()
        obj.data_balancing()

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error: {e}")
        logger.error(f"Error in line no: {er_line.tb_lineno}")

