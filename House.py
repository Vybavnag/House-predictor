from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import norm, skew
from scipy import stats
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates


load_boston = load_boston()
X = load_boston.data
y = load_boston.target

data = pd.DataFrame(X, columns=load_boston.feature_names)
data["SalePrice"] = y

compression_opts = dict(method='zip',
                        archive_name='out.csv')
data.to_csv('out.zip', index=False,
            compression=compression_opts)

data.isnull().sum()

data["SalePrice"] = np.log1p(data["SalePrice"])

sns.distplot(data['SalePrice'], fit=norm)

(mu, sigma) = norm.fit(data['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(data['SalePrice'], plot=plt)
plt.show()

plt.figure(figsize=(10, 10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.PuBu)
plt.show()

cor_target = abs(cor["SalePrice"])

relevant_features = cor_target[cor_target > 0.2]


names = [index for index, value in relevant_features.iteritems()]

names.remove('SalePrice')

X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

predictions = lr.predict(X_test)

print("Actual value of the house:- ", y_test[0])
print("Model Predicted Value:- ", predictions[0])

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(rmse)
