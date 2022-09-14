import numpy as np
import pandas as pd
import math
from scipy.optimize import least_squares
from statsmodels.nonparametric.smoothers_lowess import lowess
from prepare import PrepareData, AggregateData

MIN_OBS = 15  # Minimum observations needed to make prediction




# Constants for CasesModel - Feel free to change these
N_TRAIN = 60   # Number of observations used in training
N_SMOOTH = 15  # Number of observations used in smoothing
N_PRED = 56    # Number of new observations to predict
L_N_MIN = 5    # Number of days of exponential growth for L min boundary
L_N_MAX = 50   # Number of days of exponential growth for L max boundary

# Constants for DeathsModel - Feel free to change these
LAG = 15       # Number of days to lag cases for calculation of CFR
PERIOD = 30    # Number of days to total for CFR
last_date = None



def general_logistic_shift(x, L, x0, k, v, s):
    return (L - s) / ((1 + np.exp(-k * (x - x0))) ** (1 / v)) + s


def optimize_func(params, x, y, model):
    y_pred = model(x, *params)
    error = y - y_pred
    return error

class CasesModel:
    def __init__(self, model, data, last_date, n_train, n_smooth, 
                 n_pred, L_n_min, L_n_max, **kwargs):
        
        # Set basic attributes
        self.model = model
        self.data = data
        self.last_date = self.get_last_date(last_date)
        self.n_train = n_train
        self.n_smooth = n_smooth
        self.n_pred = n_pred
        self.L_n_min = L_n_min
        self.L_n_max = L_n_max
        self.kwargs = kwargs
        
        # Set attributes for prediction
        self.first_pred_date = pd.Timestamp(self.last_date) + pd.Timedelta("1D")
        self.pred_index = pd.date_range(start=self.first_pred_date, periods=n_pred)
        
    def get_last_date(self, last_date):
        # Use the most current date as the last actual date if not provided
        if last_date is None:
            date = self.data['original_table']['Date'].iloc[-1]
            year_month = date.strftime('%Y-%m')
            
            return year_month
        
        else:
            return pd.Timestamp(last_date)
        
    def init_dictionaries(self):
        # Create dictionaries to store results for each area
        # Executed first in `run` method
        self.smoothed = {}
        self.bounds = {}
        self.p0 = {}
        self.params = {}
        self.pred_daily = {}
        self.pred_cumulative = {}
        
        # Dictionary to hold DataFrame of actual and predicted values
        self.combined_daily = {}
        self.combined_cumulative = {}
        
        # Same as above, but stores smoothed and predicted values
        self.combined_daily_s = {}
        self.combined_cumulative_s = {}
        
    def smooth(self, s):
        s = s[:self.last_date]
        
        # not necessary to check if first value is zero
        # possible for 0 items to be recycled
        
        # if s.values[0] == 0:
        #     # Filter the data if the first value is 0
        #     last_zero_date = s[s == 0].index[-1]
        #     s = s.loc[last_zero_date:]
        #     s_daily = s.diff().dropna()
        # else:
        #     # If first value not 0, use it to fill in the 
        #     # first missing value
        #     s_daily = s.diff().fillna(s.iloc[0])

        # Don't smooth data with less than MIN_OBS values
        if len(s) < MIN_OBS:
            # return s_daily.cumsum()
            return -1

        y = s.values
        frac = self.n_smooth / len(y)
        x = np.arange(len(y))
        y_pred = lowess(y, x, frac=frac, is_sorted=True, return_sorted=False)
        s_pred = pd.Series(y_pred, index=s.index).clip(0)
        # s_pred_cumulative = s_pred.cumsum()
                
                
                
        # this is causing errors because what happens if dec value == 0
        # if s_pred_cumulative[-1] == 0:
        if s_pred[-1] == 0:
            # Don't use smoothed values if they are all 0
            # return s_daily.cumsum()
            return s_pred
        
        last_actual = s.values[-1]
        # last_smoothed = s_pred_cumulative.values[-1]
        last_smoothed = s_pred.values[-1]
        # s_pred_cumulative *= last_actual / last_smoothed
        # return s_pred_cumulative
        s_pred *= last_actual / last_smoothed
        return s_pred
    
    def get_train(self, smoothed):
        # Filter the data for the most recent to capture new waves
        return smoothed.iloc[-self.n_train:]
    
    # s = train
    def get_L_limits(self, s):
        # nonzero = np.nonzero(s)
        # last_val = s[nonzero[-1]]
        # second_last = s[nonzero[-2]]
        last_val = s[-1]
        s = s[:-1]
        while last_val == 0:
            last_val = s[-1]
            s = s[:-1]
        second_last = s[-1]
        while second_last == 0:
            s = s[:-1]
            second_last = s[-1]
        # last_pct = s.pct_change()[-1] + 1
        # this worked until marina mall
        # last_pct = nonzero.pct_change()[-1]
        last_pct =  100 * (last_val-second_last) / second_last
        # problem because last_pct < 1 => L_max < L_min
        # take note of this problem
        if last_pct >= 0:
            last_pct += 1
            L_min = last_val * last_pct ** self.L_n_min
            L_max = last_val * last_pct ** self.L_n_max 
        else:
            last_pct = 1 + last_pct
            L_max = last_val * last_pct ** self.L_n_min
            L_min = last_val * last_pct ** self.L_n_max
        # for testing purposes
        if math.isnan(L_min) == True or math.isnan(L_max) == True:
            return L_min, L_max, -100
        # if L_min and L_max is too small
        if int(L_min) == int(L_max):
            L_min = -0.1
            L_max = 0.1
        L0 = (L_max - L_min) / 2 + L_min
        
        return L_min, L_max, L0
    
    # when will L be nan?
    
    def get_bounds_p0(self, s):
        L_min, L_max, L0 = self.get_L_limits(s)
        if L0 == -100:
            return -100, -100
        x0_min, x0_max = -50, 50
        k_min, k_max = 0.01, 0.5
        v_min, v_max = 0.01, 2
        s_min, s_max = 0, s[-1] + 0.01
        s0 = s_max / 2
        lower = L_min, x0_min, k_min, v_min, s_min
        upper = L_max, x0_max, k_max, v_max, s_max
        bounds = lower, upper
        p0 = L0, 0, 0.1, 0.1, s0
        return bounds, p0
        
    def train_model(self, s, bounds, p0):
        y = s.values
        n_train = len(y)
        x = np.arange(n_train)
        res = least_squares(optimize_func, p0, args=(x, y, self.model), bounds=bounds, **self.kwargs)
        return res.x
    
    
    # start editing here

    
    def get_pred_daily(self, n_train, params):
        x_pred = np.arange(n_train - 1, n_train + self.n_pred)
        y_pred = self.model(x_pred, *params)
        # y_pred_daily = np.diff(y_pred)
        y_pred_daily = y_pred
        # return pd.Series(y_pred_daily, index=self.pred_index)
        return y_pred
    
    def get_pred_cumulative(self, s, pred_daily):
        last_actual_value = s.loc[self.last_date]
        return pred_daily.cumsum() + last_actual_value
    
    def convert_to_df(self, tables_to_access):
        # convert dictionary of areas mapped to Series to DataFrames
        self.smoothed[tables_to_access] = pd.DataFrame(self.smoothed[tables_to_access]).fillna(0).astype('int')
        self.bounds[tables_to_access] = pd.concat(self.bounds[tables_to_access].values(), 
                                    keys=self.bounds[tables_to_access].keys()).T
        self.bounds[tables_to_access].loc['L'] = self.bounds[tables_to_access].loc['L'].round()
        self.p0[tables_to_access] = pd.DataFrame(self.p0[tables_to_access], index=['L', 'x0', 'k', 'v', 's'])
        self.p0[tables_to_access].loc['L'] = self.p0[tables_to_access].loc['L'].round()
        self.params[tables_to_access] = pd.DataFrame(self.params[tables_to_access], index=['L', 'x0', 'k', 'v', 's'])
        self.pred_daily[tables_to_access] = pd.DataFrame(self.pred_daily[tables_to_access])
        self.pred_cumulative[tables_to_access] = pd.DataFrame(self.pred_cumulative[tables_to_access])
        
    def combine_actual_with_pred(self):
        for gk, df_pred in self.pred_cumulative.items():
            df_actual = self.data[gk][:self.last_date]
            df_comb = pd.concat((df_actual, df_pred))
            self.combined_cumulative[gk] = df_comb
            self.combined_daily[gk] = df_comb.diff().fillna(df_comb.iloc[0]).astype('int')
            
            df_comb_smooth = pd.concat((self.smoothed[gk], df_pred))
            self.combined_cumulative_s[gk] = df_comb_smooth
            self.combined_daily_s[gk] = df_comb_smooth.diff().fillna(df_comb.iloc[0]).astype('int')

    def run(self):
        
        self.init_dictionaries()
        category = {'original', 'item', 'collection_method', 'organisation', 'location'}
        for category in category:
            tables_to_access = f'{category}_table'
            df_cases = self.data[tables_to_access]
            self.smoothed[tables_to_access] = {}
            self.bounds[tables_to_access] = {}
            self.p0[tables_to_access] = {}
            self.params[tables_to_access] = {}
            self.pred_daily[tables_to_access] = {}
            self.pred_cumulative[tables_to_access] = {}
            
            for item, item_data in self.data[tables_to_access].items():
                # return item_data
                # scaled by 100
                # item_data *= 100
                smoothed = self.smooth(item_data)
                
                # if type(smoothed) == int:
                #     return item_data
                # jam bottle
                train = self.get_train(smoothed)
                # if item == 'MARINA MALL':
                #     return train
                   
                # complication: not all items in each cateogory
                # and not all categories will have graphs 
                # need to resolve this when plotting
                n_train = len(train)
                
                if n_train < MIN_OBS or all(item == 0 for item in train):
                    bounds = np.full((2, 5), np.nan)
                    p0 = np.full(5, np.nan)
                    params = np.full(5, np.nan)
                    pred_daily = pd.Series(np.zeros(self.n_pred), index=self.pred_index)
                            
                else:
                    # if all(item == 0 for item in train): 
                    #     return tables_to_access+item
                 
                    # else:
                    bounds, p0 = self.get_bounds_p0(train)
                    if bounds == -100 and p0 == -100:
                        return category+item
                    lower = bounds[0][0]
                    upper = bounds[1][0]
                    if lower > upper:
                        return item
                    
                    # condition 1 fixed: p0 not in range(bounds)
                    # condition 2 debugging: Each lower bound must be strictly less than each upper bound.
                    params = self.train_model(train, bounds=bounds,  p0=p0)
                    pred_daily = self.get_pred_daily(n_train, params).round(0)
                    
                # return pred_daily
                pred_cumulative = self.get_pred_cumulative(item_data, pred_daily)
                # return pred_daily

                # save results to dictionaries mapping each area to its result
                self.smoothed[tables_to_access][item] = smoothed
                self.bounds[tables_to_access][item] = pd.DataFrame(bounds, index=['lower', 'upper'], 
                                                    columns=['L', 'x0', 'k', 'v', 's'])
                self.p0[tables_to_access][item] = p0
                self.params[tables_to_access][item] = params
                self.pred_daily[tables_to_access][item] = pred_daily.astype('int')
                self.pred_cumulative[tables_to_access][item] = pred_cumulative.astype('int')
                # return pred_daily
                
        # this cannot be tables_to_access
        self.convert_to_df(tables_to_access)
        # code works till here
        self.combine_actual_with_pred()
        
    def plot_prediction(self, group, area, **kwargs):
        group_kind = f'{group}_cases'
        actual = self.data[group_kind][area]
        pred = self.pred_cumulative[group_kind][area]
        first_date = self.last_date - pd.Timedelta(self.n_train, 'D')
        last_pred_date = self.last_date + pd.Timedelta(self.n_pred, 'D')
        actual.loc[first_date:last_pred_date].plot(label='Actual', **kwargs)
        pred.plot(label='Predicted').legend()
   
data = PrepareData().run()  
df = data['original']
data_2 = AggregateData(df)
# print(data_2['item_table'])  
 
a = CasesModel(model=general_logistic_shift,
        data=data_2,
        last_date=last_date,
        n_train=N_TRAIN,
        n_smooth=N_SMOOTH,
        n_pred=N_PRED,
        L_n_min=L_N_MIN,
        L_n_max=L_N_MAX)
b = a.run()
print(b)