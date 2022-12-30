import pandas as pd
import numpy as np
import sklearn

class ExtractFeatures(object):

    def __init__(self):
        self.file_path = './XBTUSD.csv'
        self.data = pd.read_csv(self.file_path, index_col="timestamp") #reading into the dataframe
        self.data = self.data.replace('null',np.nan).fillna(0).astype('float') # replacing null values with 0s 
        self.extracted_features = None # To store the normalized features 

    
    # Each of the following functions adds a feature to the dataset for XGBoost classification

    '''Exponential Moving Average'''
    def EWMA(self, data, days):
        ema = pd.Series(pd.ewma(data['close'], span = days, min_periods = days - 1), 
        name = 'EWMA_' + str(days))
        data = data.join(ema) 
        return data

    '''Bolinger Bands'''
    def bbands(self, data, days):
        MA = data.close.rolling(window=days).mean()
        SD = data.close.rolling(window=days).std()
        data['UpperBB'] = MA + (2 * SD) 
        data['LowerBB'] = MA - (2 * SD)
        return data

    '''Commodity Channel Index'''
    def CCI(self, data, days):
        TP = (data['high'] + data['low'] + data['close']) / 3
        CCI = pd.Series((TP - pd.rolling_mean(TP, days)) / (0.015 * pd.rolling_std(TP, days)),
        name = 'CCI')
        data = data.join(CCI)
        return data

    '''Ease of Movement'''
    def EVM(self, data, days): 
        dm = ((data['high'] + data['low'])/2) - ((data['high'].shift(1) + data['low'].shift(1))/2)
        br = (data['volume'] / 100000000) / ((data['high'] - data['low']))
        EVM = dm / br 
        EVM_MA = pd.Series(pd.rolling_mean(EVM, days), name = 'EVM') 
        data = data.join(EVM_MA) 
        return data

    '''Rate of Change'''
    def ROC(self,data,days):
        N = data['close'].diff(days)
        D = data['close'].shift(days)
        roc = pd.Series(N/D,name='ROW')
        data = data.join(roc)
        return data 

    '''Force Index'''
    def ForceIndex(self, data, days): 
        FI = pd.Series(data['close'].diff(days) * data['volume'], name = 'ForceIndex') 
        data = data.join(FI) 
        return data
    
    '''Simple Moving Average'''
    def SMA(self,data, days): 
        sma = pd.Series(pd.rolling_mean(data['close'], days), name = 'SMA_' + str(days))
        data = data.join(sma) 
        return data

    def normalize(self, data):
        data = data.dropna().astype('float')
        data = sklearn.preprocessing.scale(data)
        data = pd.DataFrame(data, columns=data.columns)
        return data

