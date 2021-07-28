# Short Term Load Forecasting

Code for predicting the next-hourly load on an electrical power system given the past load and temporal data.
A vanilla neural network architecture is used to make the prediction.
Data from [UCI ML repository](https://archive.ics.uci.edu/ml/index.php) for a Portuguese Utility Elergone containing load consumption information for 370 consumers was aggregated to experiment with determining the overall system load, found under the data directory, giving an accuracy of 97.4% on test data (used as one year out of total 3-year data). An experiment was also made anomaly detection and replacement.
The system architecture was later tested on a private data for Mumbai city, giving better results (since the data size was larger.)

## Key code files 

### [Anomaly Treatment](https://github.com/sagarsj42/short-term-load-forecasting/blob/master/load-based-models/anomaly_treatment_sts.py)
Calls the Anomaly Detection algorithms of [Vector Norm](https://github.com/sagarsj42/short-term-load-forecasting/blob/master/load-based-models/vector_norm.py) and [Probability Distribution Function](https://github.com/sagarsj42/short-term-load-forecasting/blob/master/load-based-models/probability_distribution_function.py) and replaces the detected anomaly points (using withing ORing or ANDing of the outputs) with their next weekly value.

### [Data Normalization](https://github.com/sagarsj42/short-term-load-forecasting/blob/master/load-based-models/normalize-loads.py)
Normalizes the load data to lie in \[0, 1\].

### [Data Preparation](https://github.com/sagarsj42/short-term-load-forecasting/blob/master/load-based-models/prep_data.py)
Prepares the load data and data from timestamps to input vectors and output values for it to be suitable to feed a neural network.

### [Neural Network Implementation](https://github.com/sagarsj42/short-term-load-forecasting/blob/master/load-based-models/feed_back.py)
Implementation of a vanilla feedforward neural network for the regression task with capabilities for regular SGD, SGD with momentum learning with a decreasing learning rate based on training performance feedback. [This](https://github.com/sagarsj42/short-term-load-forecasting/blob/master/load-based-models/visualize_session.py) wrapper module can be called to run the process end-to-end and generate visualizations out of the training/evaluation process.

### [Visualization of forecasts](https://github.com/sagarsj42/short-term-load-forecasting/blob/master/load-based-models/visualize_forecast.py)
This module contains code for visualizing and comparing the load forecasts of a saved model.
