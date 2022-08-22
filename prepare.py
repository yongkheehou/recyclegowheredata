import numpy as np
import pandas as pd

download_link = ("https://raw.githubusercontent.com/yongkheehou/recyclegowheredata/main/random_generated_website_usage_data.csv")

class PrepareData:
    
    def __init__(self, download_new=True):
        self.download_new = download_new
        
    def download_data(self):
        df = pd.read_csv(download_link)
        return df
    
    def select_columns(self, df):
        cols = df.columns

        areas = ["Date", "Item", "House Collection/ Self Pickup", "Organisation", "Bin Location"]
        is_area = cols.isin(areas)

        filt = is_area 
        return df.loc[:, filt]
    
    def transpose_to_ts(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace = True)
        # df = df.T
        # df.index = pd.to_datetime(df.index)
        return df
    
    def group_item(self, df):
        grouping_col = df.columns[0]
        return df.groupby(grouping_col).sum()
    
    def group_collection_method(self, df):
        grouping_col = df.columns[1]
        return df.groupby(grouping_col).sum()
    
    def group_organisation(self, df):
        grouping_col = df.columns[2]
        return df.groupby(grouping_col).sum()
    
    def group_bin_location(self, df):
        grouping_col = df.columns[3]
        return df.groupby(grouping_col).sum()
    
    def run(self):
        data = {}
        if self.download_new:
            df = self.download_data()
        df = self.select_columns(df)
        df = self.transpose_to_ts(df)
        
        df1 = self.group_item(df)
        df2 = self.group_collection_method(df)
        df3 = self.group_organisation(df)
        df4 = self.group_bin_location(df)
        data['original'] = df
        data['item'] = df1
        data['collection_method'] = df2
        data['organisation'] = df3
        data['bin_location'] = df4
        
        return data

x = PrepareData().run()
print(x['original'])
# print(x['collection_method'].dtypes)


# # working code
# x = pd.read_csv(download_link)
# print(x.head)