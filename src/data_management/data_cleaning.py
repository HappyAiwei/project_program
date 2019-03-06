import matplotlib.pyplot as plt
import os
import seaborn as sns
from bld.project_paths import project_paths_join as ppj
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.style.use('ggplot')
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')

# Define some functions to handel data extraction.
def extract1(x):
    return int(str(x)[:4])

def extract2(x):
    return int(str(x)[5:7])

def extract3(x):
    return int(str(x)[8:10])

train = pd.read_csv(ppj("IN_DATA", "train.csv"))
store = pd.read_csv(ppj("IN_DATA", "store.csv"))
test = pd.read_csv(ppj("IN_DATA", "test.csv"))

# Data cleaning:
### Replace missing data, NA/NaN data with 1.
test.fillna(1, inplace=True)

### Combine marketing information with supplemental information,
### in order to offer more options in feature selection.
train = pd.merge(train, store, on='Store')
train.fillna(0, inplace=True)

### Merge supplemental information to the test dataset to make final prediction.
test = pd.merge(test, store, on='Store')
test.fillna(0, inplace=True)

### Show correlation between each pair.
correlation_map = train[train.columns].corr()
obj = np.array(correlation_map)
obj[np.tril_indices_from(obj)] = False
fig, ax = plt.subplots()
fig.set_size_inches(15, 15)
sns.heatmap(correlation_map, mask=obj, vmax=.7, square=True, annot=True)#.savefig(ppj("OUT_FIGURES", "train_store_coorr.pdf"))
plt.savefig(ppj("OUT_FIGURES", "train_store_coorr.pdf"))
#sns_plot = sns.heatmap(correlation_map, mask=obj, vmax=.7, square=True, annot=True)#.savefig(ppj("OUT_FIGURES", "train_store_coorr.pdf"))
#fig = sns_plot.get_figure()
#plt.savefig(ppj("OUT_FIGURES", "train_store_coorr.pdf"))


### Remove outlier.
train = train[train['Open'] != 0]
train = train[train['Sales'] > 0]

### Extracting Day , Month and Year as columns.
train['Year'] = train.Date.apply(extract1)
train['Month'] = train.Date.apply(extract2)
train['Day'] = train.Date.apply(extract3)
train['Date'] = train['Date'].apply(lambda x: str(x)[:7])
test['Year'] = test.Date.apply(extract1)
test['Month'] = test.Date.apply(extract2)
test['Day'] = test.Date.apply(extract3)

sns.factorplot(x='Date', y ='Sales', data=train, kind='point', aspect=2, size=12).savefig(ppj("OUT_FIGURES", "monthly_sales.pdf"))

fig, axes = plt.subplots(2, 1)
fig.set_size_inches(10, 10)
sns.barplot(x='Year', y='Sales', data=train, hue='StoreType', ax=axes[0])
sns.barplot(x='Year', y='Sales', data=train, hue='Assortment', ax=axes[1])
fig.savefig(ppj("OUT_FIGURES", "storetype_assortment_sales.pdf"))

fig, axes = plt.subplots(2,1)
fig.set_size_inches(15, 10)
sns.boxplot(x='Year', y='Sales', data=train, hue='DayOfWeek', ax=axes[0])
sns.violinplot(x='Year', y='Sales', data=train, hue='DayOfWeek', ax=axes[1])
fig.savefig(ppj("OUT_FIGURES", "dayofweeksales.pdf"))

### Split each kind of StoreType, Assortment and each year, assign 1 to corresponding data.
train = pd.get_dummies(train, columns=['StoreType', 'Assortment', 'Year'])
test = pd.get_dummies(test, columns=['StoreType', 'Assortment', 'Year'])

### Convert categorical data to numerical data.
train['StateHoliday'] = train['StateHoliday'].map({0: 0, '0': 0, 'a': 1, 'b': 1, 'c': 1})
train['StateHoliday'] = train['StateHoliday'].astype(float)
test['StateHoliday'] = test['StateHoliday'].map({0: 0, '0': 0, 'a': 1, 'b': 1, 'c': 1})
test['StateHoliday'] = test['StateHoliday'].astype(float)

test['Year_2013'] = 0
test['Year_2014'] = 0

### Log-transformed sales could narrow fluctuation,
### which is helpful to construct linear regressions.
train['log_sales'] = np.log(train['Sales'])


### Drop some insignificant variables after feature selection.
### Divide into independent variable set and dependent variable set.
X = train.drop(['Sales', 'log_sales', 'Store', 'Date', 'Customers', 'SchoolHoliday',
                'CompetitionOpenSinceYear', 'Promo2SinceYear', 'PromoInterval'], axis=1)
y = train['log_sales']

y = pd.DataFrame(y, columns=['log_sales'])

### Drop some insignificant variables
X_test = test.drop(['Id', 'Store', 'Date', 'SchoolHoliday', 'CompetitionOpenSinceYear',
                    'Promo2SinceYear', 'PromoInterval'], axis=1)



X.to_csv(ppj("OUT_DATA", "clean_X.csv"), sep=',', encoding='utf-8', index=False)
y.to_csv(ppj("OUT_DATA", "clean_y.csv"), sep=',', encoding='utf-8', index=False)
print(X_test.shape)
X_test.to_csv(ppj("OUT_DATA", "clean_X_test.csv"), sep=',', encoding='utf-8', index=False)




