import pandas as pd

file_path = 'melb_data.csv'
melb_data = pd.read_csv(file_path)
melb_data = melb_data.dropna(axis=0)
y = melb_data.Price
print (y)