import pandas as pd
# stands for extreme gradient boosting
# predict outcomes in regression fashion
# aggregate of mutliple weak prediction models
import xgboost as xgb
import matplotlib.pyplot as plt

data = pd.read_csv("all_stocks_5yr.csv")
