import numpy as np
import pandas as pd
import math
def _convert_to_vector_u(wd, wd1, ws, ws1):
    """ Converts wind speed and directions to their u vector component """
    neg_ws = -ws 
    temp2 = (math.pi/180) * wd
    temp2_1 = np.sin(temp2.iloc[:, 0].values)
    temp2_2 = np.sin(temp2.iloc[:, 1].values)
    temp2_3 = np.sin(temp2.iloc[:, 2].values)
    temp2_4 = np.sin(temp2.iloc[:, 3].values)
    df = pd.DataFrame(temp2_1, columns=['1']).join(pd.DataFrame(temp2_2, columns=['2']))
    df = df.join(pd.DataFrame(temp2_3, columns=['3']))
    df = df.join(pd.DataFrame(temp2_4, columns=['4']))
    
    neg_ws1 = -ws1 
    temp2_1 = (math.pi/180) * wd1
    temp2_1_1 = np.sin(temp2_1.iloc[:, 0].values)
    temp2_2_1 = np.sin(temp2_1.iloc[:, 1].values)
    temp2_3_1 = np.sin(temp2_1.iloc[:, 2].values)
    temp2_4_1 = np.sin(temp2_1.iloc[:, 3].values)
    df1 = pd.DataFrame(temp2_1_1, columns=['1']).join(pd.DataFrame(temp2_2_1, columns=['2']))
    df1 = df1.join(pd.DataFrame(temp2_3_1, columns=['3']))
    df1 = df1.join(pd.DataFrame(temp2_4_1, columns=['4']))
    return neg_ws.mul(df), neg_ws1.mul(df1)

def _convert_to_vector_v(wd, wd1, ws, ws1):
    """ Converts wind speed and directions to their v vector component """
    neg_ws = -ws 
    #print(wd)
    temp2 = (math.pi/180) * wd
    #print(temp2)
    temp2_1 = np.cos(temp2.iloc[:, 0].values)
    #print(temp2_1)
    temp2_2 = np.cos(temp2.iloc[:, 1].values)
    #print(temp2_2)
    temp2_3 = np.cos(temp2.iloc[:, 2].values)
    temp2_4 = np.cos(temp2.iloc[:, 3].values)
    df = pd.DataFrame(temp2_1, columns=['1']).join(pd.DataFrame(temp2_2, columns=['2']))
    df = df.join(pd.DataFrame(temp2_3, columns=['3']))
    df = df.join(pd.DataFrame(temp2_4, columns=['4']))
    
    neg_ws1 = -ws1 
    temp2_1 = (math.pi/180) * wd1
    temp2_1_1 = np.cos(temp2_1.iloc[:, 0].values)
    temp2_2_1 = np.cos(temp2_1.iloc[:, 1].values)
    temp2_3_1 = np.cos(temp2_1.iloc[:, 2].values)
    temp2_4_1 = np.cos(temp2_1.iloc[:, 3].values)
    df1 = pd.DataFrame(temp2_1_1, columns=['1']).join(pd.DataFrame(temp2_2_1, columns=['2']))
    df1 = df1.join(pd.DataFrame(temp2_3_1, columns=['3']))
    df1 = df1.join(pd.DataFrame(temp2_4_1, columns=['4']))
    return neg_ws.mul(df), neg_ws1.mul(df1)

def _convert_to_degrees(u, v):
    """ Converts u and v vector components to mathmatical degrees """ 
    result = (180/math.pi) * np.arctan2(-v, -u)
    return result

def _convert_to_met_deg(row):
    """ Converts mathatical degrees to met degrees """
    for idx in range(len(row.values)):
        if row.values[idx] == 0: row.values[idx] = 90
        elif row.values[idx] == 90: row.values[idx] = 0
        elif row.values[idx] == 180: row.values[idx] = 270
        elif row.values[idx] == 270: row.values[idx] = 180
        elif row.values[idx] > 0 and row.values[idx] < 90: 
            row.values[idx] = 90 - row.values[idx]
        elif row.values[idx] > 90 and row.values[idx] < 180: 
            row.values[idx] = 360 - (row.values[idx] - 90)
        elif row.values[idx] > 180 and row.values[idx] < 270: 
            row.values[idx] = 270 - (row.values[idx] - 180)
        else: 
            row.values[idx] = 360 - row.values[idx] + 90
    return row
    
def _convert_to_math_deg(x):
    """ Converts met degrees to mathmatical degrees """ 
    if x == 0: x = 90
    elif x == 90: x = 0
    elif x == 180: x = 270
    elif x == 270: x = 180
    elif x > 0 and x < 90: 
        x = 90 - x
    elif x > 90 and x < 180: 
        x = 270 + (180 - x) 
    elif x > 180 and x < 270: 
        x = 270 - (x - 180)
    else: 
        x = 180 - (x - 270)
    if x > 360: 
        return x - 360
    return x
    
def _convert_to_ws(u, v):
    """ Converts u and v vector to wind speed """
    return np.sqrt(u**2 + v**2)