#Data from https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
#General libraries
import pandas as pd
import numpy as np
import pickle
#Models
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
class crop_choice():
    def __init__(self):
        self.data = pd.read_csv("Crop_recommendation.csv")
        
    def dataprep(self):
        X=self.data.drop('label',axis=1)
        y=self.data['label']
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,
                                               shuffle=True,random_state=0)
        self.df = {"X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test}

    def train_choice(self, model="full"):
        """
        Train the class to choose a crop
        """
        self.dataprep()
        if model not in ["lgbm", "rf", "mlp", "full"]:
            raise ValueError("Model not available")
        elif model == "lgbm":
            res = self.train_lgbm(model_name = model)
        elif model == "rf":
            res = self.train_rf(model_name = model)
        elif model == "mlp":
            res = self.train_mlp(model_name = model)
        elif model == "full":
            self.train_lgbm(model_name = "lgbm")
            self.train_mlp(model_name = "mlp")
            res = self.train_rf(model_name = "rf")
        pickle.dump(res[1], open("model.pkl", "wb"))

    def train_mlp(self, model_name):
        model = MLPClassifier(hidden_layer_sizes=(100,50,50))
        model.fit(self.df["X_train"], self.df["y_train"])
        self.y_pred = model.predict(self.df["X_test"])
        self.performance_eval(model_name)
        return(self.performance_eval, model)

    def train_lgbm(self, model_name):
        model = lgb.LGBMClassifier()
        model.fit(self.df["X_train"], self.df["y_train"])
        self.y_pred = model.predict(self.df["X_test"])
        self.performance_eval(model_name)
        return(self.performance_eval, model)

    def train_rf(self, model_name):
        model = RandomForestClassifier()
        model.fit(self.df["X_train"], self.df["y_train"])
        self.y_pred = model.predict(self.df["X_test"])
        self.performance_eval(model_name)
        return(self.performance_eval, model)

    def performance_eval(self, model_name):
        accuracy=f1_score(self.y_pred, self.df["y_test"], average = "weighted")
        print("F1 score for {} is {}".format(model_name, accuracy))

    def choose(self, data = [90,20,30,55,66.5,6.5,120]):
        model = pickle.load(open('model.pkl', 'rb'))
        df =  pd.DataFrame(columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        df.loc[0] = data
        result = model.predict(df)
        print(result)

if __name__=="__main__":
    chooser = crop_choice()
    #chooser.train_choice()
    chooser.choose()
