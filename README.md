# Predicting Stocks Trends based on news

Stocks price is known to be fairly sensitive to news and the response on the price can be seen in the short amount of time after the news was announced. The objective of this machine learning project aims to capture and predict the trend based on news.

## Environment
* Python 3.7.3
* PyTorch 1.1.0 with CUDA support
* NumPy 1.16.4

## Running

Training new model: simply run 'main.py'
Loading existing model: import 'load_model' from 'RNN.py', which return a model and feed in a appropiate size of boolean vector (currently 10)