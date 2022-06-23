import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from statsmodels.api import qqplot
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.stattools import durbin_watson

data = pd.read_csv("MEMOIRE_PGQ.csv")
data["PQG"]=[float(elem.replace(',', '.')) for elem in data.PQG]

#Lineplot : Exploratory Data Analysis
"""
plot = sns.lineplot(data=data, x="temps", y="PQG", hue="Groupe", ci="sd")
plt.show()
"""
#QQPlot : Normality evaluation
"""
qqplot(sample.PQG)
plt.show()
for group in list(set(data.Groupe)):
    sample = data[data.Groupe == group]
    qqplot(sample.PQG)
    plt.show()
"""

###
#Regressions
###

def linear_modelling_PQG(groupe, data=data):
    """
    Linear model fitting for each group
    """
    data_1 = data[data.Groupe == groupe]
    """
    plot = sns.regplot(data=data_1, x="temps", y="PQG")
    plt.show()

    plot = sns.residplot(data=data_1, x="temps", y="PQG")
    plt.show()
    """
    print("Linear regression on group {}".format(groupe))
    #Formatting as a feature array for sklearn
    time_data=np.array(data_1.temps).reshape(-1, 1)

    # Fitting the model
    l_model = LinearRegression()
    l_model.fit(time_data, data_1.PQG)

    # Returning the R^2 for the model
    g1_r2 = l_model.score(time_data, data_1.PQG)
    print('R^2: {0}'.format(g1_r2))
    #Computes adjusted R squared
    adjusted = 1 - (1-g1_r2)*(len(data_1.PQG)-1)/(len(data_1.PQG)-time_data.shape[1]-1)
    print("Adjusted R squared. Group {} : {}".format(groupe, adjusted))

    predictions = l_model.predict(time_data) 
    df_results = pd.DataFrame({'Actual': data_1.PQG, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    #Test normality of residuals : 
    print("Normality of residuals, p-value : {}".format(shapiro(df_results.Residuals)[1]))

    #No autocorrelation of error terms
    print("Autocorrelation of residuals, Durbin-Watson : {}".format(durbin_watson(df_results.Residuals)))
    
    #Slope and intercep
    print("Slope : {}, Intercept : {}".format(l_model.coef_, l_model.intercept_))



linear_modelling_PQG(1)
linear_modelling_PQG(2)


####
#Let's use time series method because it's a time serie
###
from statsmodels.tsa.stattools import adfuller

station_test = adfuller(data.PQG, autolag = 'AIC')
print(station_test)
