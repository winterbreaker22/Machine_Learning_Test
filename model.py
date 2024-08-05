import pandas as pd

file_path = 'melb_data.csv'
melb_data = pd.read_csv(file_path)
print (melb_data.describe())