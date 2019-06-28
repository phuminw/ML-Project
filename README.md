# Predicting Stocks Trends based on news

Stocks price is known to be fairly sensitive to news and the response on the price can be seen in the short amount of time after the news was announced. The objective of this machine learning project aims to capture and predict the trend based on news.

## Environment
* Python 3.7.3
* PyTorch 1.1.0 with CUDA support
* NumPy 1.16.4

## Running

### Training new model
```
./main.py
```
Two files will be saved: one is the .model and another one is the accuracy from KFold cross validation.

### Loading existing model:

Add following lines to the script that will feed a boolean vector from TfidfVectorizer (size 10 for AMZN.model) to the model.

```
from RNN import load_model

model = load_model(PATH) // PATH to .model file

print(model(input_vector))
```