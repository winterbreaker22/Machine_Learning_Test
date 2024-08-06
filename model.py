
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

file_path = 'melb_data.csv'
melb_data = pd.read_csv(file_path)
melb_data = melb_data.dropna(axis=0)
y = melb_data.Price
print (y)

melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melb_data[melb_features]

melb_model = DecisionTreeRegressor(random_state=1)
melb_model.fit(X, y)
# print (X.head())
# print (melb_model.predict(X.head()))

predicted_home_prices = melb_model.predict(X)
# print (mean_absolute_error(y, predicted_home_prices))

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
melb_model = DecisionTreeRegressor(random_state=1)
melb_model.fit(train_X, train_y)

val_predictions = melb_model.predict(val_X)
# print(mean_absolute_error(val_y, val_predictions))

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
