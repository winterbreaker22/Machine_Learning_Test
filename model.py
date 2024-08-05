from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

file_path = 'melb_data.csv'
melb_data = pd.read_csv(file_path)
melb_data = melb_data.dropna(axis=0)
y = melb_data.Price

melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melb_data[melb_features]

melb_model = DecisionTreeRegressor(random_state=1)
melb_model.fit(X, y)
print (X.head())
print (melb_model.predict(X.head()))
