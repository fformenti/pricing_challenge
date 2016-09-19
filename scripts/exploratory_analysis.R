library(ggplot2)
library(lubridate)
library(tidyr)

#=================================
# Reading
#=================================

# ------ reading from raw files --------
# sales <- read.csv("../data/input/sales.csv", stringsAsFactors = F)
# comp_price <- read.csv("../data/input/comp_prices.csv", stringsAsFactors = F)

# ------ reading from a local database --------
# install.packages("RPostgreSQL")
require("RPostgreSQL")
source("my_pass.r")

img_path <- "../analysis/viz/"
# loads the PostgreSQL driver
drv <- dbDriver("PostgreSQL")
# creates a connection to the postgres database
con <- dbConnect(drv, dbname = "postgres",
                 host = "localhost", port = 5432,
                 user = username, password = pw)

# query the data from postgreSQL 
sales <- dbGetQuery(con, "SELECT * from b2w_schema.sales_agg")
price_def <- dbGetQuery(con, "SELECT * from b2w_schema.comp_prices where pay_type = 1")
price_im <- dbGetQuery(con, "SELECT * from b2w_schema.comp_prices where pay_type = 2")

# Removing variables
rm(con,drv,pw,username)

#=================================
# Data Treatment
#=================================
days_of_the_week <- c("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
sales$date_order <- as.Date(sales$date_order)
sales$day_week <- trimws(sales$day_week)
sales$day_week <- factor(sales$day_week, levels = days_of_the_week, ordered=TRUE)
sales$year_month = floor_date(sales$date_order, unit = c("month"))

price_im$date_order = as.Date(price_im$date_order)
price_im <- price_im[,c("prod_id","date_order","competitor","competitor_price")]

# Pulling only the minimun prices for each product per competitor on each day 
price_im <- aggregate(x = list(competitor_price = price_im$competitor_price), 
                      by = list(prod_id = price_im$prod_id, 
                                date_order = price_im$date_order,
                                competitor = price_im$competitor), 
                      FUN = min, na.rm = TRUE)

#=================================
# Exploratory Plots
#=================================

brand_green <- "#00e68a"

# ------- Times Series ---------
# Daily 
p <- ggplot(sales, aes(date_order, qty_order)) 
p <- p + theme_bw() + geom_line(colour = brand_green)
p <- p + labs(title = "Daily Sales 2015", x = "", y = "Quantity")
p <- p + facet_wrap(~prod_id, scales = "free_y")
p

ggsave("daily_sales_line.png", p, path = img_path)

p <- ggplot(sales, aes(date_order, qty_order)) 
p <- p + theme_bw() + geom_point(colour = brand_green)
p <- p + labs(title = "Daily Sales 2015 \n", x = "", y = "Quantity")
p <- p + facet_wrap(~prod_id, scales = "free_y")
p

ggsave("daily_sales_point.png", p, path = img_path)

# Monthly
monthly_sales <- aggregate(x = list(qty_order = sales$qty_order), 
                           by = list(year_month = sales$year_month, prod_id = sales$prod_id), 
                           FUN = mean, na.rm = TRUE)

p <- ggplot(monthly_sales, aes(year_month, qty_order)) + theme_bw()
p <- p+ geom_line(colour = brand_green)
p <- p + labs(title = "Monthly Sales 2015 \n", x = "", y = "Quantity")
p <- p + facet_wrap(~prod_id, scales = "free_y")
p

ggsave("monthly_sales.png", p, path = img_path)

# ------- Histograms, Densities and Boxplots ---------

# Competitor's Price BoxPlot per Product
my_title <- "Competitor's Price BoxPlot per Product \n"
p <- ggplot(price_im, aes(competitor, competitor_price)) + theme_bw() 
p <- p + geom_boxplot() + facet_wrap(~prod_id, scales = "free")
p <- p + labs(title = my_title, x = "Competitor", y = "Price")
p

ggsave("comp_boxplot.png", p, path = img_path)

# Competitor's Price Density per Product
my_title <- "Competitor's Price Density Plot per Product \n"
p <- ggplot(price_im, aes(competitor_price, fill = competitor, colour = competitor))
p <- p + geom_density(alpha = 0.1) + facet_wrap(~prod_id, scales = "free") + theme_bw()  
p <- p + labs(title = my_title, x = "Competitor's Price", y = "")
p

ggsave("comp_density.png", p, path = img_path)

# ----- Relation with Numerical Variables

# Qty sold of Product A given its price
p <- ggplot(sales, aes(price, qty_order)) + geom_point(size = 1.5)
p <- p + theme_bw()  
p <- p + labs(title = "Quantity x Price \n", x = "Price", y = "Quantity")
p <- p + facet_wrap(~prod_id, scales = "free")
p <- p + geom_smooth(se = FALSE, colour = brand_green)
p

ggsave("qty_price.png", p, path = img_path)

# Qty sold of Product A given B2W's price of Other Products
scp_df <- merge(sales, sales[,c("prod_id","date_order","price")], by = c("date_order"), all= TRUE)
my_product <- "P1"
scp_df <- scp_df[scp_df$prod_id.x == my_product,] #choosing Product A
my_title <- paste ("Quantity sold of ", my_product, "given B2W's price of Other Products \n", sep = " ")
p <- ggplot(scp_df, aes(price.y, qty_order)) + geom_point(size = 1.5)
p <- p + theme_bw()  
p <- p + labs(title = my_title, x = "Price", y = paste("Quantity sold for", my_product, sep = " "))
p <- p + facet_wrap(~prod_id.y, scales = "free") + geom_smooth(se = FALSE, colour = brand_green)
p

ggsave(paste(my_product, "qty_b2w_price.png", sep = "_"), p, path = img_path)

# Qty sold of Product A competitor's price for Product A
scp_df <- merge(sales, price_im, by = c("prod_id", "date_order"), all.y = TRUE)
my_product <- "P2"
scp_df <- scp_df[scp_df$prod_id == my_product,] #choosing Product A
my_title <- paste ("Qty sold of Product", my_product, "given Competitor's price \n", sep = " ")

p <- ggplot(scp_df, aes(competitor_price, qty_order)) + geom_point(size = 1.5)
p <- p + theme_bw()  
p <- p + labs(title = my_title, x = "Price", y = "Quantity sold")
p <- p + facet_wrap(~competitor, scales = "free") 
p

ggsave(paste(my_product, "qty_comp_price.png", sep = "_"), p, path = img_path)


# Qty sold of Product A given B2W's and competitor's price for Product A
scp_df <- merge(sales, price_im, by = c("prod_id", "date_order"), all.y = TRUE)
my_product <- "P1"
scp_df <- scp_df[scp_df$prod_id == my_product,] #choosing Product A
my_title <- paste ("Daily Sales of", my_product, "given B2W and Competitor's price \n", sep = " ")
log_qty <- FALSE

# Colored scatter plot
p <- ggplot(scp_df, aes(price, competitor_price)) + theme_bw()
if (log_qty) {
  p <- p + geom_point(aes(colour = log(qty_order))) + labs(color = "Log(Daily Sales)")
} else {
  p <- p + geom_point(aes(colour = qty_order)) + labs(color = "Daily Sales")
}
p <- p + scale_colour_gradient(low = "black", high = brand_green)
p <- p + labs(title = my_title, x = "B2W's Price", y = "Competitor's Price")
p <- p + facet_wrap(~competitor, scales = "free") 
p <- p + geom_abline(intercept = 0, slope = 1)
p

ggsave(paste(my_product, "qty_b2w_and_comp_price.png", sep = "_"), p, path = img_path)

# # Heatmap ran out of memory
# # Error:long vectors not supported yet: ../../../../R-3.2.1/src/main/memory.c:1648
# p <- ggplot(scp_df, aes(price, competitor_price, fill = qty_order)) + geom_raster()
# p <- p + theme_bw()  
# p <- p + labs(title = my_title, x = "B2W Price", y = "Comp Price")
# p <- p + facet_wrap(~competitor, scales = "free") 
# p
# 
# ggsave(paste(my_product, "qty_prices_heatmap.png", sep = "_"), p, path = img_path)

# ----- Relation with Categorical Variables
# First we will take a look at how the days of the week impact the sales of each product
weekday_sales <- aggregate(x = list(qty_order = sales$qty_order), 
                        by = list(day_week = sales$day_week, prod_id = sales$prod_id), 
                        FUN = mean, na.rm = TRUE)

my_title <-"Average Quantity Sold \n"
p <- ggplot(weekday_sales, aes(day_week, qty_order)) + theme_bw()
p <- p + geom_bar(fill = "black", stat = "identity") 
p <- p + labs(title = my_title, x = "Day of the Week", y = "Quantity Sold")
#p <- p + scale_colour_gradient(low = "black", high = brand_green)
p <- p + facet_wrap(~prod_id, scales = "free_y")
p

ggsave("dow_avg_sales.png", p, path = img_path)


#=================================
# New Features
#=================================

# Reshaping the data to perform a join
price_im_wide <- spread(price_im, competitor, competitor_price)

# Getting the competitor's prices into the sales table
df <- merge(sales, price_im_wide, by = c("prod_id", "date_order"), all.x= TRUE)

