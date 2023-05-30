import pandas as pd

class SiteIndexOE:
    
    #Note, not sure which has and doe snot have lidar
    no_lidar = []
    
    def __init__(self):
        self.index = pd.read_csv("01 One Energy Turbine Data/OneEnergyTurbineData.csv")
        
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
    
    #Do no think this is needed for One Energy
    #def aid_to_tid(self,aid):
        #return self.index[self.index['AID'] == aid]['APRS ID']
    
    def lookup_by_tid(self,tid):
        return self.index[self.index['APRS ID'] == tid].to_dict(orient='records').pop()