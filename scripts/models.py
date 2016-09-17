import sys
import psycopg2
import pandas.io.sql as psql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error

import my_pass

def reading(host, DB, username, password):
    conn_string = "host=" + host + " dbname=" + DB + " user=" + username + " password=" + password
    # get connected to the database
    connection = psycopg2.connect(conn_string)
    query = "SELECT * FROM b2w_schema.sales_agg;"
    sales = psql.read_sql(query, connection)
    # Here I'm only selecting immeadiate pay
    query = "SELECT prod_id,date_order,competitor, min(competitor_price) as competitor_price\
            FROM b2w_schema.comp_prices \
            GROUP BY prod_id,date_order,competitor,pay_type \
            HAVING pay_type = 2;"
    price_im = psql.read_sql(query, connection)
    return (sales,price_im)

class ml_models(object):

    def __init__(self, data, product):
        self.df = data
        self.product = product

    def prepare_data(self):
        # selecting a specific product to analyse
        self.df = self.df[self.df['prod_id'] == self.product]

        # Remove columns that only have missing values
        self.df = self.df.dropna(axis=1,how='all')

        # Transforming categorical varibales into factors
        le_day_week = preprocessing.LabelEncoder()
        self.df['day_week'] = le_day_week.fit_transform(self.df['day_week'])
        self.df['month'] = le_day_week.fit_transform(self.df['month'])

    def impute_values(self, cols, strategy = 'mean', axis = 0):

        imp = Imputer(missing_values='NaN', strategy = strategy, axis = axis)
        imp.fit(self.df[cols])
        self.df[cols] = imp.transform(self.df[cols])

    def price_diff(self, comp_price = ['C1','C2','C3','C4','C5','C6']):
        self.df['C1'] = self.df['price'] - self.df['C1']
        self.df['C2'] = self.df['price'] - self.df['C2']
        self.df['C3'] = self.df['price'] - self.df['C3']
        self.df['C4'] = self.df['price'] - self.df['C4']
        self.df['C5'] = self.df['price'] - self.df['C5']
        self.df['C6'] = self.df['price'] - self.df['C6']

    def make_train_test(self,frac = 0.8, random_state = 200):
        # creating a training and test sets
        self.df_train = self.df.sample(frac = frac, random_state = random_state)
        self.df_test = self.df.drop(self.df_train.index)

    def select_features(self, keep_feat = [], drop_feat = []):

        # Isolate Response variable
        self.Y_train, self.Y_test = self.df_train['qty_order'], self.df_test['qty_order']

        # warning: drop_feat overrides keep_feat !!!
        if drop_feat:
            self.X_train = self.df_train.drop(drop_feat, axis=1, inplace=False)
            self.X_test = self.df_test.drop(drop_feat, axis=1, inplace=False)
        elif keep_feat:
            self.X_train = self.df_train[keep_feat]
            self.X_test = self.df_test[keep_feat]

        # Remove Response variable from regressor set
        self.X_train.drop('qty_order', axis=1, inplace=True)
        self.X_test.drop('qty_order', axis=1, inplace=True)

        # Total number of regressors
        self.n_regressors = len(self.X_test.columns)


    def predict_test(self, print_mse = True):
        # Predict and update dataset
        self.Y_pred = self.clf.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, self.Y_pred)
        if print_mse:
            print "MSE: %.4f" % mean_squared_error(self.Y_test, self.Y_pred)


class GBM(ml_models):
    # Gradient Boosting
    def __init__(self, data, product):
        ml_models.__init__(self, data, product)

    def fit_gb(self, params):
        # Fit model
        print self.X_train.columns
        self.clf = GradientBoostingRegressor(**params)
        self.clf.fit(self.X_train, self.Y_train)

    def plot_feature_importance(self, n):
        importances = self.clf.feature_importances_
        feature_names = self.X_test.columns
        indices = np.argsort(importances)[::-1][:n]
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(10,6)
        plt.title("Feature importances", fontsize = 16)
        xlabels = [feature_names[int(i)] for i in indices]
        plt.bar(range(n), importances[indices],
                color="#799DBB",  align="center")
        plt.grid()
        plt.xticks(range(n), xlabels, rotation=90)
        plt.xlim([-1, n])
        plt.ylim([0, min(1, max(importances[indices]+0.0005))])
        plt.xlabel('Features', fontsize = 14)
        plt.ylabel('Feature Importance', fontsize = 14)
        plt.title('Product '+  self.product +' Variable Importance')
        plt.show()

class MLR(ml_models):
    # Multiple linear regression
    def __init__(self, data, product):
        ml_models.__init__(self, data, product)

    def fit_mlr(self):
        # Fit model
        self.clf = linear_model.LinearRegression(fit_intercept=False)
        self.clf.fit(self.X_train, self.Y_train)
        print self.clf.coef_

if __name__ == "__main__":

    # User inputs
    host = 'localhost'
    DB = 'postgres'
    username = my_pass.username
    password = my_pass.password

    # Reading tables
    sales, price_im = reading(host, DB, username, password)

    # Reshaping the data to perform a join
    price_im_wide = pd.pivot_table(price_im, index = ['prod_id','date_order'], columns = ['competitor'], values = 'competitor_price')
    price_im_wide.reset_index(inplace = True)
    df = pd.merge(sales, price_im_wide, how='left', on=['prod_id','date_order'])


    ##############################
    ### Fitting Models for P2
    ##############################

    # ------- Gradient Boosting -----------
    gb_p2 = GBM(df, 'P2') 
    gb_p2.prepare_data() 
    gb_p2.impute_values(cols = ['C1','C2','C3','C4','C5','C6'])
    gb_p2.price_diff()
    gb_p2.make_train_test()
    gb_p2.select_features(drop_feat = ['prod_id','revenue','date_order'])
    
    params = {'n_estimators': 1000, 'max_depth': 2, 'max_features': 'sqrt', 'random_state': 5}
    gb_p2.fit_gb(params)

    # Predict
    gb_p2.predict_test()

    # Plot feature importance
    gb_p2.plot_feature_importance(gb_p2.n_regressors)


    # ------- Multiple Linear Regression -----------
    mlr_p2 = MLR(df, 'P2')
    mlr_p2.prepare_data() 
    mlr_p2.impute_values(cols = ['C1','C2','C3','C4','C5','C6'])
    mlr_p2.price_diff()
    mlr_p2.make_train_test()
    mlr_p2.select_features(drop_feat = ['prod_id','revenue','date_order'])
    mlr_p2.fit_mlr()

    # Predict
    mlr_p2.predict_test()










