#! /usr/bin/env python3

import pandas as pd
from datetime import datetime
import datetime as dt
import numpy as np

def main():
    # For label change in price
    def findStockChange(row, dataset, dateOffset = 0):
        currentstockDay = None
        date = datetime.strptime(row, '%Y-%m-%d')
        date = date + dt.timedelta(days=dateOffset)
        row = date.strftime('%Y-%m-%d')
        currentstockDay = dataset[dataset['Date'] == row]
        if not currentstockDay.empty:
            return currentstockDay.iloc[0]['Close'] > currentstockDay.iloc[0]['Open']
        else:
            return False


    news = pd.read_csv("../Data/stocknews/uci-news-aggregator.csv")
    price = pd.read_csv('../Data/stockprice/Stocks/msft.us.txt')
    tech_news = news[news['CATEGORY'] == 't']

    tech_news['TIMESTAMP'] = tech_news['TIMESTAMP'].map(lambda x: datetime.fromtimestamp(int(int(x)/1000)).strftime('%Y-%m-%d'))
    price = price[(price['Date'] > '2014-03-09') & (price['Date'] < '2014-08-29')]

    tech_news['today'] = tech_news['TIMESTAMP'].apply(lambda row: findStockChange(row, price)).astype(np.int32)
    tech_news['tomorrow'] = tech_news['TIMESTAMP'].apply(lambda row: findStockChange(row, price, dateOffset=1)).astype(np.int32)
    
    return tech_news