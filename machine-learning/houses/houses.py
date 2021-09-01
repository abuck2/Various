#general packages
import numpy as np
import pandas as pd
import time, sys, os

#data exploration
import seaborn as sns
import matplotlib.pyplot as plt


#dataprep
from sklearn.preprocessing import StandardScaler

#models
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


class HousePredictor:

    def __init__(self, explore = False, model = "RF"):
        self.explore = explore
        
        if model in ["GB", "RF"]:
            self.model = model
        else : 
            raise ValueError("Model should be GB or RF")


    def get_data(self, path):
        self.data = pd.read_csv(path+"train.csv")
        self.test = pd.read_csv(path+"test.csv")
        self.test_basic = self.test

    def exploration(self):
        print(self.data.shape)
        print(self.data.columns)
        sns.distplot(self.data['SalePrice']);
        
        #skewness and kurtosis
        print("Skewness: %f" % self.data['SalePrice'].skew())
        print("Kurtosis: %f" % self.data['SalePrice'].kurt())
        
        #correlation matrix
        corrmat = self.data.corr()
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True);


    def dataprep(self):
        print("Preparing the data...")

        #missing data
        total = self.data.isnull().sum().sort_values(ascending=False)
        percent = (self.data.isnull().sum()/self.data.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        missing_data.head(20)

        #dealing with missing data
        """
        ## Better ways to do this...
        self.data = self.data.drop((missing_data[missing_data['Total'] > 1]).index,1)
        self.data = self.data.drop(self.data.loc[self.data['Electrical'].isnull()].index)
        self.data.isnull().sum().max() #just checking that there's no missing data missing...
        """
        self.data.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count'])
        self.data["LotAreaCut"] = pd.qcut(self.data.LotArea,10)
        self.data['LotFrontage'] = self.data\
                .groupby(['LotAreaCut','Neighborhood'])['LotFrontage']\
                .transform(lambda x: x.fillna(x.median()))
        
        self.data['LotFrontage'] = self.data\
                .groupby(['LotAreaCut'])['LotFrontage']\
                .transform(lambda x: x.fillna(x.median()))
        
        cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
        for col in cols:
            self.data[col].fillna(0, inplace=True)

        cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
        for col in cols1:
            self.data[col].fillna("None", inplace=True)

        # fill in with mode
        cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
        for col in cols2:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        #numerical to categorical woooyeah
        NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
        for col in NumStr:
            self.data[col]=self.data[col].astype(str)

        #get numericals clumns
        train_numericCol = self.data.select_dtypes(include=[np.number]).columns.values
        
        #In case we forgot some columns 
        #self.data.fillna(self.data.mean(),inplace = True)
        #self.test.fillna(self.test.mean(),inplace = True)

        #standardizing data (check later if any improment)
        #self.data = StandardScaler().fit_transform(self.data['SalePrice'][:,np.newaxis]);

    def train(self):
        #connection
        h2o.init(ip="localhost", port=54321)

        ## Convert data into h2o Frame
        train = h2o.H2OFrame(self.data)
        test = h2o.H2OFrame(self.test)
        test_original = h2o.H2OFrame(self.test_basic)
        # Split the train dataset
        train, valid, test = train.split_frame(ratios=[0.7, 0.15], seed=42)
        self.test = test
        self.test_original = test_original

        # Seperate the target data and store it into y variable
        y = 'SalePrice'
        Id = test['Id']

        # remove target and Id column from the dataset and store rest of the columns in X variable
        X = list(train.columns)
        X.remove(y)
        X.remove('Id')
        
        if self.model == "GB":
            self.build_modelGB(train, test, valid, X, y)
        elif self.model == "RF":
            self.build_modelRF(train, test, valid, X, y)

    def build_modelGB(self, train, test, valid, X, y):
        # Prepare the hyperparameters
        gbm_params = {
                    'learn_rate': [0.01, 0.1], 
                    'max_depth': [4, 5, 7],
                    'sample_rate': [0.6, 0.8],               # Row sample rate
                    'col_sample_rate': [0.2, 0.5, 0.9]       # Column sample rate per split (from 0.0 to 1.0)
            
                    }

        # Prepare the grid object
        gbm_grid = H2OGridSearch(model=H2OGradientBoostingEstimator,   # Model to be trained
                        grid_id='gbm_grid1',                  # Grid Search ID
                        hyper_params=gbm_params,              # Dictionary of parameters
                        search_criteria={"strategy": "Cartesian"}   # RandomDiscrete
                    )

        # Train the Model
        start = time.time() 
        gbm_grid.train(x=X,y=y, 
                        training_frame=train,
                        validation_frame=valid,
                        ntrees=100,      # Specify other GBM parameters not in grid
                        score_tree_interval=5,     # For early stopping
                        stopping_rounds=3,         # For early stopping
                        stopping_tolerance=0.0005,
                        seed=1)

        end = time.time()
        (end - start)/60


        # Find the Model grid performance 
        gbm_gridperf = gbm_grid.get_grid(sort_by='RMSE',decreasing = False)
        print(gbm_gridperf)

        # Identify the best model generated with least error
        best_gbm_model = gbm_gridperf.models[0]
        self.predictor = best_gbm_model

    def build_modelRF(self, train, test, valid, X, y):
        
        # Prepare the hyperparameters
        nfolds = 5
        rf_params = {
                    'max_depth': [3, 4,5],
                    'sample_rate': [0.8, 1.0],               # Row sample rate
                    'mtries' : [2,4,3]
                    }

        # Search criteria for parameter space
        search_criteria = {'strategy': "RandomDiscrete",
                            "seed": 1,
                            'stopping_metric': "AUTO",
                            'stopping_tolerance': 0.0005
                            }
        
        # Prepare the grid object
        rf_grid = H2OGridSearch(model=H2ORandomForestEstimator,   # Model to be trained
                        grid_id='rf_grid',                  # Grid Search ID
                        hyper_params=rf_params,              # Dictionary of parameters
                        search_criteria=search_criteria,   # RandomDiscrete
                        )

        # Train the Model
        start = time.time() 
        rf_grid.train(x=X,y=y, 
                    training_frame=train,
                    validation_frame=valid,
                    ntrees=100,      
                    score_each_iteration=True,
                    nfolds = nfolds,
                    fold_assignment= "Modulo",
                    seed=1
                    )

        end = time.time()
        (end - start)/60
        
        # Find the Model performance 
        rf_gridperf = rf_grid.get_grid(sort_by='RMSE',decreasing = False)
        rf_gridperf

        # Identify the best model generated with least error
        best_rf_model = rf_gridperf.models[0]
        self.predictor = best_rf_model

    def prediction(self):
        perfs = self.predictor.model_performance(self.test)
        print(perfs.gini)
        
        results = self.predictor.predict(self.test_original).as_data_frame()
        
        sub = pd.DataFrame()
        sub['Id'] = results.index + 1461
        sub['SalePrice'] = results
        #sub.head()
        sub.to_csv('houses_prediction_'+str(self.model)+'.csv', index=False)
        
    def run(self, path):
        self.get_data(path)

        if self.explore == True :
            self.exploration()

        self.dataprep()
        self.train()
        self.prediction()


if __name__ == "__main__":
    path = "/home/alexis/python_files/houses/"
    regressor = HousePredictor(explore = False, model = "RF")
    regressor.run(path)
