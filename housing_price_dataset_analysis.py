import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# load the data
data = pd.read_csv("Housing.csv")
# print(data)
# print(data.info())
# To drop the null values
data.dropna(inplace=True)
# print(data.info())

# convert the string data into numeric data
string_data = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
data[string_data] = data[string_data].apply(lambda x: x.map({"yes": 1, "no": 0}))
# print(data)
# convert furnishing status variable into numeric using one hot encoding
status = pd.get_dummies(data["furnishingstatus"], drop_first=True)
# print(status.head())
print("******************************************")
# concat the status variable with the training data
data = pd.concat([data, status], axis=1)
# print(data)
data = data.drop("furnishingstatus", axis=1)
# split data into train and test
X = data.drop("price", axis=1)
y = data["price"]
# print(X)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
training_data = pd.concat([X_train, y_train], axis=1)
print("training data",training_data)
# Find the histogram of the training data
training_data.hist(figsize=(20, 15))
plt.show()

# scale the data
scaler = MinMaxScaler()
num_vars = ["area", "bedrooms", "bathrooms", "stories", "parking", "price"]
training_data[num_vars] = scaler.fit_transform(training_data[num_vars])
print(training_data)

# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(training_data.corr(), annot = True, cmap="YlGnBu")
plt.show()
# Lets scatter plot the variables
sns.relplot(
    data=training_data, x="price", y="area",
    col="stories", hue="mainroad", style="mainroad",
    kind="scatter"
)
plt.show()

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# fit the model
reg.fit(X_train, y_train)
# predict the model
print(reg.score(X_test, y_test))
print("******************************************")

# Let's use the random forest regressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))


# Let's use the grid search to find the best parameters
params = {
    "n_estimators": [int(x) for x in range(50,200)],
    "max_features": [4,5,6,7,8],
    # "max_depth": [int(x) for x in range(3, 10)],
    "max_leaf_nodes": [int(x) for x in range(3, 6)],

}

grid = GridSearchCV(rf, param_grid=params, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)
best_model = grid.best_estimator_
print(best_model.score(X_test, y_test))






