from scikit-learn import DecisionTreeRegressor

melb_data = DecisionTreeRegressor(random_state=1)
print (melb_data)

melb_data.fit(X, y)
print (melb_data)