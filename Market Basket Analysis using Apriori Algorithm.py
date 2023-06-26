#Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Reading the Dataset
dataset = pd.read_csv("online_retail.csv")
print(dataset)
#Data Exploration
print("The number of rows and columns in the dataset")
print(dataset.shape)
print("Description of the data")
print(dataset.describe())
print("Information about the dataset")
print(dataset.info())
print("Finding the null and non-null values of the columns in the dataset")
print(dataset.notnull().sum())
print(dataset.isnull().sum())
print("Finding the NaN values")
print(dataset.isna().sum())
print("The first five rows ")
print(dataset.head())
print(dataset["Description"].value_counts())
X = dataset["Description"].value_counts().sort_values(ascending=False)[:20]
print(X)
#Setting index as Date
dataset.set_index('InvoiceDate',inplace = True)
#Converting date into a particular format
dataset.index = pd.to_datetime(dataset.index)
#Gathering information about products
total_item = len(dataset)
total_days = len(np.unique(dataset.index.date))
total_months = len(np.unique(dataset.index.year))
print(total_item,total_days,total_months)
#Visulaizing the most bought items/products
plt.figure(figsize=(15, 10))
Barplot = sns.barplot(x=X.index, y=X.values)
Barplot.set_xticklabels(labels=["WHITE HANGING HEART T-LIGHT HOLDER", "REGENCY CAKESTAND 3 TIER", "JUMBO BAG RED RETROSPOT", "PARTY BUNTING", "LUNCH BAG RED RETROSPOT", "ASSORTED COLOUR BIRD ORNAMENT", "SET OF 3 CAKE TINS PANTRY DESIGN", "PACK OF 72 RETROSPOT CAKE CASES", "LUNCH BAG  BLACK SKULL.", "NATURAL SLATE HEART CHALKBOARD","POSTAGE","JUMBO BAG PINK POLKADOT","HEART OF WICKER SMALL","JAM MAKING SET WITH JARS","JUMBO STORAGE BAG SUKI","PAPER CHAIN KIT 50'S CHRISTMAS","JUMBO SHOPPER VINTAGE RED PAISLEY","LUNCH BAG CARS BLUE","LUNCH BAG SPACEBOY DESIGN","JAM MAKING SET PRINTED"], rotation=102)
Barplot.set(xlabel="Item Name", ylabel="Quantity Bought")
Barplot.set_title("TOP 20 MOST BOUGHT ITEMS/PRODUCTS", fontdict={'size': 30, 'weight': 'bold'})
plt.show()

#Consolidating the customer's ID along with each total item/product bought
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

transactions = dataset.groupby(["CustomerID", "Description"])["Quantity"].sum().unstack().reset_index().set_index("CustomerID")
transactions = transactions.fillna(0)
print(transactions)
#Encoding
def encode(x):
    if x <= 0:
        return 0
    elif x>= 0:
        return 1
market_basket = transactions.applymap(encode)
print(market_basket)
#Importing mlxtend and tabulate modules
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from tabulate import tabulate
#Implementing the apriori algorithm
frequent_itemset = apriori(market_basket.astype('bool'),min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemset, metric='lift', min_threshold=1)
#Getting the results
print(rules.to_markdown())



