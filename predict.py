import math
import quandl
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
import pickle
from statistics import mean
import linear_regression
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import r_square

quandl.ApiConfig.api_key = 'nuYEr_cSKvuJjstbzSzV'


df = quandl.get("WIKI/TSLA")
#df = pd.read_csv('tesla.csv')


df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#print(df.head())


forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
print(df['label'].tail())
#df.to_csv('out.csv')


#print(df[['Adj. Close','Adj. Open']].values.tolist())
#print(list(df['label'].values.tolist()))


X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
print(df['label'].tail())
y = np.array(df['label'])
# we do not need to segment the last part of y because it is empty

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


clf = LinearRegression()

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

print('confidence = ',confidence)


forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


def apply_linear_regression(xs,ys):
    m, b = linear_regression.best_fit_slope_and_intercept(xs,ys)
    regression_line = [(m*x)+b for x in xs]
    r_squared = r_square.coefficient_of_determination(y,regression_line)

    return regression_line,r_squared

x = np.array(df['Adj. Close'])
x = x[:-forecast_out]

regression_line,r_squared = apply_linear_regression(x,y)

print('r squared = ',r_squared)

print('Current FutureExpect FuturePredict')
for i in range(len(regression_line)):
  print('{} {} {}'.format(x[i],y[i],regression_line[i]))


plt.scatter(np.array(df['label'][:-20]),x)
plt.plot(regression_line,x,color='y')
plt.xlabel('Current Value')
plt.ylabel('Predicted Value')
plt.show()

