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

from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import my_pass


def reading(host, DB, username, password):
    conn_string = "host=" + host + " dbname=" + DB + " user=" + username + " password=" + password
    # get connected to the database
    connection = psycopg2.connect(conn_string)
    query = "SELECT * FROM b2w_schema.sales_agg;"
    sales = psql.read_sql(query, connection)
    # Here I'm only selecting immeadiate pay
    query = "SELECT prod_id,date_order,competitor, min(competitor_price) as competitor_price            FROM b2w_schema.comp_prices             GROUP BY prod_id,date_order,competitor,pay_type             HAVING pay_type = 2;"
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
    
    def qty_lag_1(self):
        # Quantity sold from previous day
        self.df.sort(['date_order'], ascending=[1], inplace = True)
        self.df['qty_lag_1'] = self.df['qty_order'].shift(1)
        self.df.loc[0,('qty_lag_1')] = self.df.loc[1,('qty_lag_1')]
    
    def cumsum_lag_3(self):
        pass
    
    
    def delta_lag2_lag1(self):
        pass
    
    
    def rem_rows_comp_NA(self, cols):
        na_rows = self.df[cols].isnull().sum(axis=1)
        self.df = self.df.ix[na_rows != len(cols)]
        
    def fill_comp_price_NA(self, col, cols):
        df = self.df[cols]
        df.drop(col, axis=1, inplace=True)
        self.df.loc[:,('aux_col')] = df.mean(axis=1, skipna = True)
        null_rows = self.df.loc[:,(col)].isnull()
        self.df.loc[null_rows,(col)] = self.df.loc[null_rows,('aux_col')]
        self.df.drop('aux_col', axis=1, inplace=True)
    
    def fill_prod_price_NA(self,col,min_value):
        null_rows = self.df.loc[:,(col)].isnull()
        self.df.loc[null_rows,(col)] = min_value
    
    def make_train_test(self,frac = 0.8, random_state = 200):
        # creating a training and test sets
        self.df_train = self.df.sample(frac = frac, random_state = random_state)
        self.df_test = self.df.drop(self.df_train.index)
        
    def select_X_Y(self,Y, X_keep = [], X_drop = []):
        
        # Isolate Response variable
        self.Y_train, self.Y_test = self.df_train[Y], self.df_test[Y]

        # warning: drop_feat overrides keep_feat !!!
        if X_drop:
            self.X_train = self.df_train.drop(X_drop, axis=1, inplace=False)
            self.X_test = self.df_test.drop(X_drop, axis=1, inplace=False)
            # Remove Response variable from regressor set
            self.X_train.drop(Y, axis=1, inplace=True)
            self.X_test.drop(Y, axis=1, inplace=True)
        elif X_keep:
            self.X_train = self.df_train[X_keep]
            self.X_test = self.df_test[X_keep]

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

    # # Analysis

    # Output Path
    output_path = '/Users/felipeformentiferreira/Documents/github_portfolio/pricing_challenge/data/output/'

    # ### Reading Data
    # User inputs
    host = 'localhost'
    DB = 'postgres'
    username = my_pass.username
    password = my_pass.password

    # Reading tables
    sales, price_im = reading(host, DB, username, password)

    # ### Getting the Competitor's Price

    # Reshaping the data to perform a join in order to get Competitor's Prices
    price_im_wide = pd.pivot_table(price_im, index = ['prod_id','date_order'], columns = ['competitor'], values = 'competitor_price')
    price_im_wide.reset_index(inplace = True)
    df = pd.merge(sales, price_im_wide, how='left', on=['prod_id','date_order'])

    # ## Fitting Models

    # ### P1

    # #### Linear Regression (Baseline)
    mlr = MLR(df, 'P1')
    mlr.prepare_data() 
    mlr.make_train_test()
    ans_df_rhs = mlr.df[['prod_id','date_order']]
    mlr.select_X_Y(Y = 'qty_order', X_keep = ['price'])
    mlr.fit_mlr()

    # Predict
    mlr.predict_test()


    # Writing the results to output folder
    d = {'price': mlr.X_test['price'], 'Y_test': mlr.Y_test, 'Y_pred': mlr.Y_pred}
    ans_df = pd.DataFrame(data=d)
    ans_df = pd.merge(ans_df, ans_df_rhs, how='left', left_index = True, right_index = True)
    ans_df.to_csv(output_path + 'mlr_' + my_product + '.csv',index = False)

    print ans_df[['prod_id','date_order','price','Y_test','Y_pred']].head(10)

    # #### Gradient Boosting

    my_product = 'P1'
    competitors = ['C1','C2','C3','C5','C6']

    # Creating a Gradient Boosting Object with the dataframe and a given product
    gb = GBM(df, my_product) 
    gb.prepare_data()

    # Removing rows where there is no price for any competitor
    gb.rem_rows_comp_NA(competitors)
    for comp in competitors:
        gb.fill_comp_price_NA(comp,competitors)

    # Creating a training and a test set to evaluate model later
    gb.make_train_test()
    ans_df_rhs = gb.df[['prod_id','date_order']]
    gb.select_X_Y(Y = 'qty_order', X_drop = ['prod_id','revenue','date_order'])

    # Fit Model
    params = {'n_estimators': 1000, 'max_depth': 2, 'max_features': 'sqrt', 'random_state': 5}
    gb.fit_gb(params)

    # Predict
    gb.predict_test()

    # Writing the results to output folder
    d = {'price': gb.X_test['price'], 'Y_test': gb.Y_test, 'Y_pred': gb.Y_pred}
    ans_df = pd.DataFrame(data=d)
    ans_df = pd.merge(ans_df, ans_df_rhs, how='left', left_index = True, right_index = True)
    ans_df.to_csv(output_path + my_product + '.csv',index = False)

    # Plot feature importance
    gb.plot_feature_importance(gb.n_regressors)

    print ans_df[['prod_id','date_order','price','Y_test','Y_pred']].head(10)


    # ### P2

    my_product = 'P2'
    competitors = ['C1','C2','C3','C4','C5','C6']

    # Creating a Gradient Boosting Object with the dataframe and a given product
    gb = GBM(df, my_product) 
    gb.prepare_data()

    # Removing rows where there is no price for any competitor
    gb.rem_rows_comp_NA(competitors)
    for comp in competitors:
        gb.fill_comp_price_NA(comp,competitors)

    # Creating a training and a test set to evaluate model later
    gb.make_train_test()
    gb.select_X_Y(Y = 'qty_order', X_drop = ['prod_id','revenue','date_order'])

    # Fit Model
    params = {'n_estimators': 1000, 'max_depth': 2, 'max_features': 'sqrt', 'random_state': 5}
    gb.fit_gb(params)

    # Predict
    gb.predict_test()

    # Writing the results to output folder
    d = {'price': gb.X_test['price'], 'Y_test': gb.Y_test, 'Y_pred': gb.Y_pred}
    ans_df = pd.DataFrame(data=d)
    ans_df.to_csv(output_path + my_product + '.csv')

    # Plot feature importance
    gb.plot_feature_importance(gb.n_regressors)


    # ### P3

    my_product = 'P3'
    competitors = ['C1','C2','C3','C4','C5','C6']

    # Creating a Gradient Boosting Object with the dataframe and a given product
    gb = GBM(df, my_product) 
    gb.prepare_data()

    # Removing rows where there is no price for any competitor
    gb.rem_rows_comp_NA(competitors)
    for comp in competitors:
        gb.fill_comp_price_NA(comp,competitors)

    # Creating a training and a test set to evaluate model later
    gb.make_train_test()
    gb.select_X_Y(Y = 'qty_order', X_drop = ['prod_id','revenue','date_order'])

    # Fit Model
    params = {'n_estimators': 1000, 'max_depth': 2, 'max_features': 'sqrt', 'random_state': 5}
    gb.fit_gb(params)

    # Predict
    gb.predict_test()

    # Writing the results to output folder
    d = {'price': gb.X_test['price'], 'Y_test': gb.Y_test, 'Y_pred': gb.Y_pred}
    ans_df = pd.DataFrame(data=d)
    ans_df.to_csv(output_path + my_product + '.csv')

    # Plot feature importance
    gb.plot_feature_importance(gb.n_regressors)





