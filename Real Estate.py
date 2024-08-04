import pandas as pd
import numpy as np
housing = pd.read_csv("data.csv")
housing.head()
housing.info()
housing['CHAS'].value_counts()
housing.describe()

# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing= strat_train_set.copy()

corr_matrix = housing.corr()

corr_matrix['MEDV'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["RM","ZN","MEDV","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))

housing.plot(kind="scatter", x = "RM", y="MEDV", alpha=0.8)

housing["TAXRM"] = housing ["TAX"]/ housing["RM"]

housing.head()

corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

housing.plot(kind="scatter", x = "TAXRM", y="MEDV", alpha=0.8)

housing=strat_train_set.drop('MEDV', axis=1)
housing_labels = strat_train_set["MEDV"].copy()

housing.describe()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
imputer.fit(housing)

imputer.statistics_

X = imputer.transform(housing)

housing_tr = pd.DataFrame(X, columns=housing.columns)

housing_tr.describe()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipline = Pipeline([
    ('imputer', SimpleImputer(strategy ='median')),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipline.fit_transform(housing)

housing_num_tr.shape

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]

some_lables = housing_labels.iloc[:5]

prepared_data= my_pipline.transform(some_data)

model.predict(prepared_data)

list(some_lables)

from sklearn.metrics import mean_squared_error
housing_predictions =  model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse =  np.sqrt(mse)

rmse

from sklearn.model_selection import cross_val_score
scores =  cross_val_score(model, housing_num_tr,housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

rmse_scores

def print_scores (scores):
    print("scores:", scores)
    print("mean",scores.mean())
    print("standard_deviation", scores.std())

print_scores(rmse_scores)

from joblib import dump,load
dump(model,"Real_estate.joblib")

