import json
import ast
import psycopg2, psycopg2.extras

DB_DSN = "host=nosqlproject.cbrmqgj6aclu.us-west-2.rds.amazonaws.com dbname=dbproject user=xxxxx password=xxxxxx"

SALES = '../data/input/sales.csv'
COMP_PRICES = '../data/input/comp_price.csv'

def read_sales(filename):
  """
  transforms a file with json into tuples
  returns: tuples
  """
  data = list()
  try:
      f = open(filename)
      for line in f:
        try:
          data.append(line)
        except:
          pass
      f.close()
  except Exception as e:
      print e

  return data

def read_comp_price(filename):
  """
  transforms a file with json into tuples
  returns: tuples
  """
  data = list()
  try:
      f = open(filename)
      for line in f:
        try:
          data.append(line)
        except:
          pass
      f.close()
  except Exception as e:
      print e

  return data

def create_table(query):

  try:
     conn = psycopg2.connect(dsn=DB_DSN)
     cur = conn.cursor()
     cur.execute(query)
     conn.commit()
  except psycopg2.Error as e:
     print e.message
  else:
     cur.close()
     conn.close()

def insert_sales_data(data):

  try:
     sql = "INSERT INTO sales VALUES(%s, %s, %s, %s, %s, %s, %s, %s)"
     conn = psycopg2.connect(dsn=DB_DSN)
     cur = conn.cursor()
     cur.executemany(sql, data)
     conn.commit()
  except psycopg2.Error as e:
     print e.message
  else:
     cur.close()
     conn.close()

def insert_comp_price_data(data):

  try:
     sql = "INSERT INTO com_price VALUES(%s, %s, %s, %s, %s, %s, %s)"
     conn = psycopg2.connect(dsn=DB_DSN)
     cur = conn.cursor()
     cur.executemany(sql, data)
     conn.commit()
  except psycopg2.Error as e:
     print e.message
  else:
     cur.close()
     conn.close()

def drop_table(my_table):
    
  try:
     sql = "DROP TABLE IF EXISTS " + my_table + ";"
     conn = psycopg2.connect(dsn=DB_DSN)
     cur = conn.cursor()
     cur.execute(sql)
     conn.commit()
  except psycopg2.Error as e:
     print e.message
  else:
     cur.close()
     conn.close()

if __name__ == '__main__':

  print "******* dropping sales table **********"
  drop_table('sales')

  print "******* creating table sales **********"
  sql = "create table b2w_schema.sales(\
             PROD_ID char(2)\
           , DATE_ORDER char(10)\
           , QTY_ORDER float\
           , REVENUE float);"

  create_table(sql)

  print "******* reading sales data **********"
  sales_data = read_sales(SALES)

  print "******* inserting data into sales table **********"
  insert_sales_data(sales_data)

  print "******* dropping comp_prices table **********"
  drop_table('comp_prices')

  print "******* creating table com_prices **********"
  sql = "create table b2w_schema.comp_prices(\
       PROD_ID char(10)\
       , DATE_EXTRACTION char(19)\
       , COMPETITOR char(2)\
       , COMPETITOR_PRICE float\
       , PAY_TYPE int) ;"
  create_table(sql)

  print "******* reading comp_prices data **********"
  comp_prices_data = transform_meta_data(COMP_PRICES)

  print "******* inserting data into comp_prices table **********"
  insert_com_price_data(comp_prices_data)
