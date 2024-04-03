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

open_price = 'open'
close_price = 'close'

model = xgb.XGBRegressor()
model.fit(train_data[open_price], train_data[close_price])

predictions = model.predict(test_data[open_price])
accuracy = model.score(test_data[open_price], test_data[close_price])

print(f'Accuracy of the model: {accuracy * 100}%')


plt.plot(data[close_price], label="Close price")
plt.plot(test_data[close_price].index, predictions, label='Predictions')
plt.legend()
plt.show()