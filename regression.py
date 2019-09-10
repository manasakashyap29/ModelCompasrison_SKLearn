import pandas as pd
import pprint
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


df = pd.read_excel('Real estate valuation data set.xlsx', header = 0)
df = df.set_index('No')

#Separating Attributes and Target Variable
df_x = df.iloc[:, 0:6]
df_y = df.iloc[:, -1]

#Scaling Attributes
df_x = preprocessing.StandardScaler().fit_transform(df_x)

#15% Test data
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.15, random_state = 123)

mse_all = []

#List of models
models = [LinearRegression(), Ridge(), Lasso(), ElasticNet(), SGDRegressor(), KNeighborsRegressor(), DecisionTreeRegressor(), SVR(), RandomForestRegressor()]

for model in models:

    model.fit(x_train,y_train)
    results = model.predict(x_test)

    #Calculate Mean Squared Error
    mse = mean_squared_error(results, y_test.to_numpy())
    mse_all.append((type(model).__name__, mse))


mse_all = pd.DataFrame(mse_all, columns=['Model', 'Mean Squared Error'])
print(mse_all)