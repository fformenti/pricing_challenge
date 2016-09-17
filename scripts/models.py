import sys
import psycopg2
import pandas.io.sql as psql
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
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

class GBM:        
    def __init__(self, data, product):
        self.df = data
        self.product = product

    def prepare_data(self, regressors = [], frac = 0.8, random_state = 200):

		# selecting a specific product to analyse
		self.df = self.df[self.df['prod_id'] == self.product]

		# creating a training and test sets
		self.df_train = self.df.sample(frac = frac, random_state = random_state)
		self.df_test = self.df.drop(self.df_train.index)

		# Selecting the proper regressors
		self.Y_train, self.Y_test = self.df_train['qty_order'], self.df_test['qty_order']
		if regressors:
			self.X_train, self.X_test = self.df_train[regressors], self.df_test[regressors]
		else:
			self.X_train = self.df_train.drop('qty_order', axis=1, inplace=False)
			self.X_test = self.df_test.drop('qty_order', axis=1, inplace=False)			

		# Remove columns that only have missing values
		self.X_train = self.X_train.dropna(axis=1,how='all')
		self.X_test = self.X_test.dropna(axis=1,how='all')

		# Transforming categorical to binaries
		#self.X_train = pd.get_dummies(self.X_train, prefix=['day_week'])
		#self.X_test = pd.get_dummies(self.X_test, prefix=['day_week'])

		# Transforming categorical varibales into factors
		le_day_week = preprocessing.LabelEncoder()
		self.X_train['day_week'] = le_day_week.fit_transform(self.X_train['day_week'])
		self.X_test['day_week'] = le_day_week.fit_transform(self.X_test['day_week'])

		# Total number of regressors
		self.n_regressors = len(self.X_test.columns)

    def fit_gb(self, params):
        # Fit model
        self.clf = GradientBoostingRegressor(**params)
        self.clf.fit(self.X_train, self.Y_train)

    def predict_test(self, print_mse = True):
        # Predict and update dataset
        self.Y_pred = self.clf.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, self.Y_pred)
        if print_mse:
            print "MSE: %.4f" % mean_squared_error(self.Y_test, self.Y_pred)

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

if __name__ == "__main__":

	# User inputs
	host = 'localhost' 
	DB = 'postgres'
	username = my_pass.username
	password = my_pass.password

	regressors = ['price','day_week','month']
	product = 'P2'

	# Reading tables
	sales, price_im = reading(host, DB, username, password)

	# Reshaping the data to perform a join
	price_im_wide = pd.pivot_table(price_im, index = ['prod_id','date_order'], columns = ['competitor'], values = 'competitor_price')
	price_im_wide.reset_index(inplace = True)
	df = pd.merge(sales, price_im_wide, how='left', on=['prod_id','date_order'])

	# Classifying
	model = GBM(df, product) # throw error in case of wrong input
	model.prepare_data(regressors) # throw error in case of wrong input
	params = {'n_estimators': 1000, 'max_depth': 5, 'max_features': 'sqrt', 'random_state': 5}
	model.fit_gb(params)

	# Predict and create CSV file
	model.predict_test()

	# Plot feature importance
	model.plot_feature_importance(model.n_regressors)







