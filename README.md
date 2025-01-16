# JS-Kaggle-2025
Jane Street Real-Time Market Data Forecasting 2025: https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting

dataset could also be downloaded through this site.

## List of resources
1. Data Visualization: this notebook ([data-visualization notebook](data-visualization.ipynb)) will help you understand the meaning of each data column
2. Exploratory Data Analysis: this notebook ([EDA notebook](EDA.ipynb)) gives us several insights of how to use the data.
    - lag data is essential for reponder_6 prediction
    - some strong correlation patterns exist between other responders and responder_6
    - this time series data does not have the characteristics of stationary and is difficult to fit
    - use R2 as objective function to train model
    - ensemble model could improve prediction accuracy
    - due to that there are a lot of new data coming in, online learning is inevitable
3. Ensemble model: this notebook ([ensemble model notebook](lgb-xgb-and-catboost.ipynb)) uses lgb, xgb, and catboost, and train on lag data. Note that we do not use online learning, but the R2 (0.0064) is acceptable.
