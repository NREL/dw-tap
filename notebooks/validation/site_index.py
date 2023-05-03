import pandas as pd

class SiteIndex:
    
    no_lidar = ["t007", "t074"]
    
    def __init__(self):
        self.index = pd.read_csv("01 Bergey Turbine Data/bergey_sites.csv")
        
    def tids(self,having_lidar=False):
        selected = self.index["APRS ID"].tolist()
        if having_lidar:
            return [x for x in selected if not(x in self.no_lidar)]
        else:
            return selected
        
    def head(self):
        return self.index.head()
    
    def itertuples(self):
        return self.index.itertuples()