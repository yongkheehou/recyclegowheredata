import numpy as np
import pandas as pd

download_link = ("https://raw.githubusercontent.com/yongkheehou/recyclegowheredata/main/random_generated_website_usage_data.csv")

class PrepareData:
    
    def __init__(self, download_new=True):
        self.download_new = download_new
        
    def download_data(self):
        df = pd.read_csv(download_link)
        return df
    
    def write_data(self, data, directory, **kwargs):
        for name, df in data.items():
            df.to_csv(f"{directory}/{name}.csv", **kwargs)
            
    def read_local_data(self, group, kind):
        name = f"{group}_{kind}"
        return pd.read_csv(f"data/raw/{name}.csv")

# working code
x = pd.read_csv(download_link)
print(x.head)