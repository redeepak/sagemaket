from itertools import count
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")



def preprocess(car):
    m=np.mean(car["tax"])
    car.fillna(m,inplace=True)
    car.describe
    car.duplicated().sum()
    car.drop_duplicates(inplace=True)

    car.model=car.model.astype("object")
    car.year=car.year.astype("object")
    car.transmission=car.transmission.astype("object")
    car.fuelType=car.fuelType.astype("object")
    car.engineSize=car.engineSize.astype("object")


    car.replace([np.inf,-np.inf],np.nan,inplace=True)
    car.dropna(axis=0,how="any",inplace=True)


    car["price"]=np.log1p(car["price"])
    car["mileage"]=np.log1p(car["mileage"])

    le_model=LabelEncoder()
    car["model"]=le_model.fit_transform(car["model"])

    le_trans=LabelEncoder()
    car["transmission"]=le_trans.fit_transform(car["transmission"])

    le_size=LabelEncoder()
    car["engineSize"]=le_size.fit_transform(car["engineSize"])

    le_fuel=LabelEncoder()
    car["fuelType"]=le_fuel.fit_transform(car["fuelType"])
    return car


train=preprocess(train)
test=preprocess(test)

x_train=train.drop("price",axis=1).values
y_train=train["price"].values

x_test=test.drop("price",axis=1).values
y_test = test['price'].values


lgr=LinearRegression(fit_intercept=True)
lgr.fit(x_train,y_train)
y_pred=np.expm1(lgr.predict(x_test))


# I added this code to fulfil the 2nd reqirement i.e., Return the csv file test.csv containing the prediction column which is price in this case

df = pd.read_csv("test.csv")
df["Pred_Price"] = pd.Series(y_pred)
df.to_csv("test.csv", index=True)
print(df)
