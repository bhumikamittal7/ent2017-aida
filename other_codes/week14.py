#Loading the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Reading the dataset
data = pd.read_csv("week14.csv")
data.head()

data.describe()

sns.heatmap(data[['Height', 'Weight']].corr(), annot=True, fmt=".2g")
plt.tight_layout()

X = data['Height']
Y = data['Weight']

#doing the regression
X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()
predictions = results.predict(X)
results.params

#preddicting based on model
weight_predicted = results.predict(X)
weight_predicted

#computing the error
data['pred'] = weight_predicted
data['error'] = Y - weight_predicted
data.head()

#residual plot
plt.scatter(data['Height'], data['error'])
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

#check the normality of the error
from scipy.stats import shapiro
shapiro(np.abs(data['error']))