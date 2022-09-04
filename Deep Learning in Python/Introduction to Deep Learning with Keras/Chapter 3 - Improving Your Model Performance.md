## chapter 3-1

Defining Estimators

```python
# Define the model and set the number of steps with DNNRegressor
model = estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[2,2])
model.train(input_fn, steps=1)

# Define the model and set the number of steps with LinearRegressor
model = estimator.LinearRegressor()(feature_columns=feature_list)
model.train(input_fn, steps=2)

```