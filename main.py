import pandas as pd
# stands for extreme gradient boosting
# predict outcomes in regression fashion
# aggregate of mutliple weak prediction models
import xgboost as xgb
import matplotlib.pyplot as plt
import os

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "all_stocks_5yr.csv"))

# train with 80% of the data and test on remaining(20%) of the data
train_data = data.iloc[:int(0.8 * len(data)), :]
test_data = data.iloc[int(0.8 * len(data)):, :]


# since our model is a regression model(simple regression model), 
# we need to create a variable(feature), from which the model will make predictions
open_price = 'open'
# target varibale for training and measureing accuracy of the model 
close_price = 'close'

model = xgb.XGBRegressor()
# train the model with previously create d train data
model.fit(train_data[open_price], train_data[close_price])

# make predicitons based on "open_price" feature
predictions = model.predict(test_data[open_price])
accuracy = model.score(test_data[open_price], test_data[close_price])

print(f'Accuracy of the model: {accuracy * 100}%')

# plot the original data and predicted data using pyplot
plt.plot(data[close_price], label="Close price")
plt.plot(test_data[close_price].index, predictions, label='Predictions')
plt.legend()
plt.show()