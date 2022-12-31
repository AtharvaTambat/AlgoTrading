import pandas as pd
import numpy as np
from sklearn import preprocessing

class ExtractFeatures(object):

    def __init__(self):
        self.file_path = './XBTUSD.csv'
        self.data = pd.read_csv(self.file_path, index_col="timestamp") #reading into the dataframe
        self.data = self.data.replace('null',np.nan).fillna(0).astype('float') # replacing null values with 0s 
        self.extracted_features = None # To store the normalized features 

    
    # Each of the following functions adds a feature to the dataset for XGBoost classification

    '''Exponential Moving Average'''
    def EWMA(self,days):
        ema = pd.Series(self.data['close'].ewm(span = days).mean(), 
        name = 'EWMA_' + str(days))
        self.data = self.data.join(ema) 

    '''Bolinger Bands'''
    def bbands(self, days):
        MA = self.data['close'].rolling(window=days, min_periods=1).mean()
        SD = self.data['close'].rolling(window=days, min_periods=1).std()
        self.data['UpperBB'] = MA + (2 * SD) 
        self.data['LowerBB'] = MA - (2 * SD)

    '''Commodity Channel Index'''
    def CCI(self, days):
        TP = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        CCI = pd.Series((TP - TP.rolling(window=days, min_periods=1).mean()) / (0.015 * TP.rolling(window=days, min_periods=1).std()),
        name = 'CCI')
        self.data = self.data.join(CCI)

    '''Ease of Movement'''
    def EVM(self, days): 
        dm = ((self.data['high'] + self.data['low'])/2) - ((self.data['high'].shift(1) + self.data['low'].shift(1))/2)
        br = (self.data['volume']) / ((self.data['high'] - self.data['low']))
        EVM = dm / br 
        EVM_MA = pd.Series(EVM.rolling(window=days, min_periods=1).mean(), name = 'EVM') 
        self.data = self.data.join(EVM_MA) 

    '''Rate of Change'''
    def ROC(self,days):
        N = self.data['close'].diff(days)
        D = self.data['close'].shift(days)
        roc = pd.Series(N/D,name='ROW')
        self.data = self.data.join(roc)

    '''Force Index'''
    def ForceIndex(self,days): 
        FI = pd.Series(self.data['close'].diff(days) * self.data['volume'], name = 'ForceIndex') 
        self.data = self.data.join(FI) 
    
    '''Simple Moving Average'''
    def SMA(self, days): 
        sma = pd.Series(self.data['close'].rolling(window=days, min_periods=1).mean(), name = 'SMA_' + str(days))
        self.data = self.data.join(sma) 

    def normalize(self):
        cols = self.data.columns
        self.data = self.data.dropna().astype('float')
        self.data = preprocessing.scale(self.data)
        self.data = pd.DataFrame(self.data, columns=cols)

    def true_values(self):
        X = self.data.values
        ind = list(self.data.columns).index('open')
        y = []
        for i in range(X.shape[0]-1):
            if (X[i+1,ind]-X[i,ind])>0:
                y.append(1)
            else:
                y.append(0)
        y = np.array(y)
        X = X[:-1]
        return X,y

    def split_train_test(self,X,y):
        split_ratio=0.8
        train_size = int(round(split_ratio * X.shape[0]))
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train, X_test, y_train, y_test

    
    


