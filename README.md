# JS-Kaggle-2025
Jane Street Real-Time Market Data Forecasting 2025: https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting (dataset could also be downloaded through this site)

We tentatively achieved Top 1% on the Leaderboard, looking forward to the result on June!

## Collection of Insights
1. Data Visualization: this notebook ([data-visualization notebook](data-visualization.ipynb)) will help you understand the meaning of each data column
2. Exploratory Data Analysis: this notebook ([EDA notebook](EDA.ipynb)) gives us several insights of how to use the data.
    - lag data is essential for reponder_6 prediction
    - some strong correlation patterns exist between other responders and responder_6
    - this time series data does not have the characteristics of stationary and is difficult to fit
    - use R2 as objective function to train model
    - ensemble model could improve prediction accuracy
    - due to that there are a lot of new data coming in, online learning is inevitable
    - While some competitors advised against using the 'weight' field during training to avoid potential bias, our empirical results showed that incorporating weights as input features and implementing a weighted R2 loss function improved performance. 
3. Ensemble model: As an initial experiment, we tested an ensemble of tree-based models (LGB, XGB, and CatBoost) ([ensemble model notebook](lgb-xgb-and-catboost.ipynb)), which is trained on lag data. While this approach didn't incorporate online learning, it achieved a reasonable R2 of 0.0064.
4. We show major stuff of our approach, like how we use the data, model architecture, model training, objective function used (refer to [model train file](all_train_weight.py)), but make some parameters anonymized.
    - Online learning becomes crucial when prediction data arrives in daily updates and model updates are permitted. Based on our experiments, this approach improved the R2 score by approximately 0.002 on the hidden set. 
    - We implemented a three-stage training pipeline, which proved to be an effective practice for handling financial time series data:
        - `Warm-up Training`: Using shuffled data, large batches, and low learning rates in the initial phase, we observed convergence to a best epoch within 5-6 epochs. Validation results showed minimal R2 improvements beyond this point, suggesting efficient model initialization.
        - `Adjust-Training`: This phase simulates online learning conditions, building upon the warm-up model. Data arrives sequentially by date_id, mixed with a portion of shuffled historical data. Validation results demonstrated improved model performance compared to the warm-up phase.
        - `Online Learning`: This final stage represents the live evaluation phase on Kaggle, where the model updates with each new date_id of data, maintaining consistency with the adjust-training methodology.
    
    - Some critical insights about three-stage training are:
        - The `Warm-up Training` phase proves essential for preventing overfitting in subsequent `Adjust-Training`. Without this initial stabilization phase, the model tends to overfit to recent patterns in the sequential data. This suggests that establishing a robust baseline understanding of general market patterns is crucial before introducing temporal dynamics.
        - Maintaining consistent training methodologies and data organization between `Adjust-Training` and `Online Learning` phases. 
        - During `Online Learning`, we recommend reducing the learning rate and including historical data to prevent catastrophic forgetting, a common challenge in sequential financial data processing where new patterns might override important historical learning.
        - The effectiveness of `adjust-training` indicates strong short-term temporal dependencies in the financial data. 

5. While sequence models theoretically offer a higher ceiling for this problem due to their ability to capture temporal dependencies in financial data, our implementation in [seq_model](seq_transformer.py) encountered severe overfitting issues. These challenges could potentially be addressed through refinements in data organization, model architecture, and scale adjustments. However, due to time constraints (joining the competition just 3 weeks before the deadline), we chose to stick with our well-optimized non-sequence model. 
