import numpy as np
import logging
import sys
from scipy.stats import yeojohnson, boxcox
from logging_code import setup_logging

logger = setup_logging('var_tran')


def vt_outliers(x_train_num, x_test_num):
    try:
        x_train = x_train_num.copy()
        x_test = x_test_num.copy()

        logger.info(f'Before Train Columns: {x_train.columns}')
        logger.info(f'Before Test Columns: {x_test.columns}')


        # monthly charges using Yeo-Johnson

        if 'MonthlyCharges' in x_train.columns:
            x_train['MonthlyCharges_yeo'], lam = yeojohnson(x_train['MonthlyCharges'])
            x_test['MonthlyCharges_yeo'] = yeojohnson(x_test['MonthlyCharges'], lmbda=lam)

            x_train.drop('MonthlyCharges', axis=1, inplace=True)
            x_test.drop('MonthlyCharges', axis=1, inplace=True)


        # TOtalCharges using BoxCox (ONLY if positive)

        if 'TotalCharges' in x_train.columns:
            # make positive (important!)
            x_train['TotalCharges'] = x_train['TotalCharges'].clip(lower=1)
            x_test['TotalCharges'] = x_test['TotalCharges'].clip(lower=1)

            x_train['TotalCharges_boxcox'], lam = boxcox(x_train['TotalCharges'])
            x_test['TotalCharges_boxcox'] = boxcox(x_test['TotalCharges'], lmbda=lam)

            x_train.drop('TotalCharges', axis=1, inplace=True)
            x_test.drop('TotalCharges', axis=1, inplace=True)


        #  Tenure column using sqrt
        if 'tenure' in x_train.columns:
            x_train['tenure_sqrt'] = np.sqrt(x_train['tenure'])
            x_test['tenure_sqrt'] = np.sqrt(x_test['tenure'])

            x_train.drop('tenure', axis=1, inplace=True)
            x_test.drop('tenure', axis=1, inplace=True)


        # IQR Trimming Function

        def iqr_trim(train_col, test_col):
            q1 = train_col.quantile(0.25)
            q3 = train_col.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            train_trim = np.clip(train_col, lower, upper)
            test_trim = np.clip(test_col, lower, upper)

            return train_trim, test_trim



        for col in list(x_train.columns):
            if col.endswith('_yeo') or col.endswith('_boxcox') or col.endswith('_sqrt'):
                x_train[col + '_trim'], x_test[col + '_trim'] = iqr_trim(
                    x_train[col], x_test[col]
                )

                x_train.drop(col, axis=1, inplace=True)
                x_test.drop(col, axis=1, inplace=True)

        logger.info(f'After Train Columns: {x_train.columns}')
        logger.info(f'After Test Columns: {x_test.columns}')

        return x_train, x_test

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error: {e}")
        logger.error(f"Error in line no: {er_line.tb_lineno}")