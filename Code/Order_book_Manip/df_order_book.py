import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def order_book():

    df_init_msft = pd.read_csv(r'data\2008-01-02-MSFT.O-bbo.csv.gz')
    df_init_msft['time'] = pd.to_datetime(df_init_msft.time)
    df_init_msft.set_index(df_init_msft.time)
    df_init_msft.set_index('time', inplace=True)
    df_init_msft = df_init_msft[df_init_msft.index.hour > 11]

    df_yhoo_init = pd.read_csv(r'data\2008-01-02-YHOO.OQ-bbo.csv.gz')
    df_yhoo_init['time'] = pd.to_datetime(df_yhoo_init.time)
    df_yhoo_init.set_index(df_yhoo_init.time)
    df_yhoo_init.set_index('time', inplace=True)
    df_yhoo_init = df_yhoo_init[100:]

    df_matrix = df_init_msft.join(df_yhoo_init, how='outer',
                                  lsuffix=('-msft'), rsuffix='-yhoo')



    df_matrix = df_matrix.fillna(method='ffill')

    df_matrix = df_matrix.drop_duplicates()

    df_matrix = df_matrix[50:]

    df_matrix = df_matrix[:-100]

    df_logdiff = np.log(df_matrix) - np.log(df_matrix.shift(1))

    df_logdiff = df_logdiff.replace(0, np.nan)

    cols_to_fill = df_logdiff.columns.tolist()
    cols_to_fill.remove('bid-price-yhoo')

    df_logdiff[cols_to_fill] = df_logdiff[cols_to_fill].fillna(method='ffill')

    df_logdiff['bid-price-yhoo'] = df_logdiff['bid-price-yhoo'].fillna(0)

    df_logdiff = df_logdiff[df_logdiff['bid-price-yhoo'] != 0]

    prediction_bid_price_yhoo = df_logdiff['bid-price-yhoo'].shift(-1)

    df_logdiff['prediction-bid-price-yhoo'] = prediction_bid_price_yhoo

    df_logdiff = df_logdiff.dropna()
    df_logdiff

    df_order_books = df_logdiff

    df_order_books['price_increase'] = (df_order_books['bid-price-yhoo'] < df_order_books['prediction-bid-price-yhoo']).astype(int)

    df_order_books.drop('prediction-bid-price-yhoo', axis=1, inplace=True)

    return(df_order_books)