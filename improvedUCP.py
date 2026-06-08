import pandas as pd
import numpy as np

#import data
df = pd.read_csv("used car price/used_cars.csv")

#drop unneccessary column
df.drop(['model','engine','ext_col','int_col','accident','clean_title'],axis = 1,inplace = True)

#filling most repeated to null values
df['fuel_type'] = df['fuel_type'].fillna(df['fuel_type'].mode()[0])

#changing datatypes to int/float
df['milage'] = df['milage'].str.replace(',','',regex = False).str.replace('mi.','',regex = False).astype(float)
df['price'] = df['price'].str.replace('$','',regex = False).str.replace(',','',regex = False).astype(float)

print(df.info)

#changing objects using dummies
category_col = [
    'brand',
    'fuel_type',
    'transmission'
]
df = pd.get_dummies(df,columns = category_col,drop_first = True)

#feature engineering
current_year = 2026
df['model_year'] = current_year - df['model_year']

#declaring x and y
x = df.drop('price', axis=1)
feature_names = x.columns
x = x.astype(float).values
y = df['price'].values

#normalize feature
x_mean = np.mean(x,axis = 0)
x_std = np.std(x,axis = 0)
x = (x - x_mean)/x_std

#train and test
split = int(0.8 * len(x))
x_train = x[:split]
y_train = y[:split]

x_test = x[split:]
y_test = y[split:]

#multiple linear regression
class MultipleLinearRegression:
    def __init__(self,learning_rate = 0.001,epoches = 1000):
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.w = None
        self.b = 0
    def predict(self,x):
        return np.dot(x,self.w) + self.b
    def compute_loss(self,y_pred,y_true):
        return np.mean((y_pred - y_true)**2)
    def fit(self,x,y_true):
        n_samples,n_features = x.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.epoches):
            y_pred = self.predict(x)
            dw = (2/n_samples) * np.dot(x.T,(y_pred - y_true))
            db = (2/n_samples) * np.sum(y_pred - y_true)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if i%100 == 0:
                loss = self.compute_loss(y_pred,y_true)
                print(f"epoches{i} and loss:{loss:.4f}")

print(x.shape)

model = MultipleLinearRegression(learning_rate = 0.001,epoches = 1000)

model.fit(x_train,y_train)

new_car = pd.DataFrame({
    'car_age': [10],
    'milage': [4000],
    'brand': ['Toyota'],
    'fuel_type': ['Gasoline'],
    'transmission': ['Automatic']

})
new_car = pd.get_dummies(new_car,columns = category_col,drop_first = True)
new_car = new_car.reindex(columns = feature_names,fill_value = 0)

new_car = (new_car - x_mean) / x_std

predicted_price = model.predict( new_car)

#evaluating
predictions = model.predict(x_test)

mse = np.mean((y_test - predictions)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - predictions))

print(x.shape)
print("MAE :", mae)
print("RMSE:", rmse)

#evaluating
predictions = model.predict(x_test)

ss_total = np.sum((y_test - np.mean(y_test))**2)
ss_res = np.sum((y_test - predictions)**2)
r2 = 1 - (ss_res / ss_total)
print("R2 Score:", r2)

print("\nPredicted Price:",predicted_price)