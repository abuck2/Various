#Source data : https://www.kaggle.com/census/homeownership-rate-time-series-collection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima.model import ARIMA
# exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class GiniPrediction:
    def __init__(self):
        self.initial_data = pd.read_csv("gini/GINIALLRF.csv")
        self.initial_data["years_passed"] = list(range(0, self.initial_data.shape[0]))
        self.initial_data["datetime"] = [datetime.strptime(elem, '%Y-%m-%d') for elem in self.initial_data["date"]]
        #self.initial_data = self.initial_data.set_index('datetime').asfreq('D')

        self.data = self.initial_data.iloc[:65]
        self.test = self.initial_data.iloc[65:]

    def data_exploration(self):
        print(self.data.columns)
        plt.figure(figsize=(16,5), dpi=100)
        plt.plot(self.data.date,self.data.value, color='tab:red')
        plt.xticks(color='w')
        plt.gca().set(title="Income inequality", xlabel="Date", ylabel="Gini coeff.")
        #plt.show()
        plt.savefig("img/giniplot.png")

        


    def modelling(self, model = "HW"):
        if model == "lm":
            self.linear_model()
        elif model == "arima":
            self.arima_model()
        elif model == "HW":
            self.hw_model()

    def linear_model(self):
        model = smf.ols('value ~ years_passed', data=self.data)
        model = model.fit()
        print(model.summary())
        # Predict values
        predictions = model.predict()        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['years_passed'], self.data['value'], 'o')           # scatter plot showing actual data
        plt.plot(self.data['years_passed'], predictions, 'r', linewidth=2)   # regression line
        plt.xlabel('Time')
        plt.ylabel('Gini coeff')
        plt.title('Linear regression on gini coefficient')
        plt.savefig("img/lm_statsmodel.png")

        #Out of sample predictions
        predictions_data = model.predict(self.test["years_passed"])
        #predictions_data = pd.DataFrame(data=[predictions[0:10], future_time], columns=["years_passed", "prediction"])
        mse = self.mean_squared_e(predictions_data, self.test.value)
        print("MSE LR = {}".format(mse))
        
        sns.lmplot(x='years_passed',y='value',data=self.data,fit_reg=True)
        plt.scatter(x=self.test["years_passed"], y=predictions_data, color='r')
        plt.scatter(x=self.test["years_passed"], y=self.test["value"], color='g')
        plt.savefig("img/lmplot.png")

        

    def arima_model(self):
        #result = seasonal_decompose(self.data.value, model='additive')
        result = adfuller(self.data.value)
        print('Aself.data Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
	        print('\t%s: %.3f' % (key, value))

        # Original Series
        fig, axes = plt.subplots(3, 2, sharex=True)
        axes[0, 0].plot(self.data.value); axes[0, 0].set_title('Original Series')
        plot_acf(self.data.value, ax=axes[0, 1])

        # 1st Differencing
        axes[1, 0].plot(self.data.value.diff()); axes[1, 0].set_title('1st Order Differencing')
        plot_acf(self.data.value.diff().dropna(), ax=axes[1, 1])

        # 2nd Differencing
        axes[2, 0].plot(self.data.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
        plot_acf(self.data.value.diff().diff().dropna(), ax=axes[2, 1])

        plt.savefig("img/acf_plot.png")

        ## Aself.data Test
        print(ndiffs(self.data.value, test='adf'))  #1

        # KPSS test
        print(ndiffs(self.data.value, test='kpss'))  #2

        # PP test:
        print(ndiffs(self.data.value, test='pp')) #1

        # PACF plot of 1st differenced series
        plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

        fig, axes = plt.subplots(1, 2, sharex=True)
        axes[0].plot(self.data.value.diff()); axes[0].set_title('1st Differencing')
        axes[1].set(ylim=(0,5))
        plot_pacf(self.data.value.diff().dropna(), ax=axes[1])

        plt.savefig("img/pacf_plot.png")

        model = ARIMA(self.data.value, order=(1,1,1))
        model_fit = model.fit()
        print(model_fit.summary())

 
        #Out of sample predictions
        predictions = model_fit.forecast(len(self.test.value))
        mse = self.mean_squared_e(predictions, self.test.value)
        print("MSE ARIMA = {}".format(mse))

        
        
        predictions_in = model_fit.predict(start = 1, end = len(self.data.value))
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['years_passed'], self.data['value'], 'o')           # scatter plot showing actual data
        plt.plot(self.data['years_passed'], predictions_in, 'r', linewidth=2)   # regression line
        plt.scatter(x=self.test["years_passed"], y=predictions, color='r')
        plt.scatter(x=self.test["years_passed"], y=self.test["value"], color='g')
        plt.xlabel('Time')
        plt.ylabel('Gini coeff')
        plt.title('ARIMA regression on gini coefficient')
        plt.show()
        plt.savefig("img/arima_statsmodel.png")

        

        


    def hw_model(self):
        df = self.data
        df.sort_index(inplace=True)
        df = df.set_index('datetime')
        decompose_result = seasonal_decompose(df.value.asfreq('YS'),model="additive")
        decompose_result.plot()
        plt.savefig("img/decompose.png")

        model = ExponentialSmoothing(self.data.value)
        model_fit = model.fit()

        print(model_fit.summary())

        #Out of sample predictions
        predictions = model_fit.forecast(len(self.test.value))
        mse = self.mean_squared_e(predictions, self.test.value)
        print("MSE HW = {}".format(mse))
        print(predictions)

    

        self.data["HW"] = model_fit.fittedvalues
        self.data[["value", "HW"]].plot(title="test")
        plt.show()

        test_predictions = model_fit.forecast(len(self.test.value))
        self.data["value"].plot(legend=True,label="TRAIN")
        self.test["value"].plot(legend=True,label="TEST",figsize=(6,4))
        test_predictions.plot(legend=True,label="PREDICTION")
        plt.title("Train, Test and Predicted Test using Holt Winters")     
        plt.savefig("img/hw_statsmodel.png")




    def mean_squared_e(self, pred, real):
        if len(pred) != len(real):
            raise ValueError("Predicted and test set should have the same length")
        
        
        sq_e = (pred-real)**2
        mse = sum(sq_e)/len(sq_e)

        return mse



if __name__=="__main__":
    predictor = GiniPrediction()
    #predictor.data_exploration()
    predictor.modelling()
