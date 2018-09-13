from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
from urllib.request import urlopen
from azure.storage.blob import BlockBlobService, PublicAccess
import hashlib

def hashme(s):
    return int(hashlib.sha1(s).hexdigest(), 16) % (10 ** 8)

# Get Data


delays_data = urlopen(delayswithairporturl)
weather_data = urlopen(weatherwithairporturl)

delays = pd.read_csv(delays_data)
weather = pd.read_csv(weather_data, dtype={'Visibility': np.str, 'WindSpeed': np.str, 'SeaLevelPressure': np.str, 'HourlyPrecip': np.str})

# standardize time
delays['CRSDepHour'] = np.floor(delays['CRSDepTime']/100)
weather['Hour'] = np.ceil(weather['Time']/100)

# adjust windspeed
windspeed = weather['WindSpeed']
windspeed = windspeed.replace("M", 0.005)
weather['WindSpeed'] = pd.to_numeric(windspeed)

# adjust SeaLevelPressure
pressure = weather['SeaLevelPressure']
pressure = pressure.replace("M", 29.92)
weather['SeaLevelPressure'] = pd.to_numeric(pressure)

# adjust SeaLevelPressure
precip = weather['HourlyPrecip']
precip = precip.replace("T", 0.005)
precip.fillna(0.005, inplace=True)
weather['HourlyPrecip'] = pd.to_numeric(precip)

# pare down columns
weather = weather[['AirportCode', 'Month', 'Day', 'Hour', 'WindSpeed', 'SeaLevelPressure', 'HourlyPrecip']]
delays = delays.drop(['OriginLatitude', 'OriginLongitude', 'DestLatitude', 'DestLongitude', 'CRSDepTime', 'DepDelay', 'CRSArrTime', 'OriginAirportName', 'DestAirportName', 'ArrDelay', 'ArrDel15', 'Cancelled'], axis=1)

# join data frames
finaldata = pd.merge(delays, weather, 'inner',
         left_on=['OriginAirportCode', 'Month', 'DayofMonth', 'CRSDepHour'],
         right_on=['AirportCode', 'Month', 'Day', 'Hour'])
finaldata = finaldata.drop(['AirportCode', 'Day', 'Hour', 'Year'], axis=1)

# hash strings to integers
finaldata['OrigAirportHash'] = finaldata['OriginAirportCode'].str.encode("UTF-8").apply(hashme)
finaldata['DestAirportHash'] = finaldata['DestAirportCode'].str.encode("UTF-8").apply(hashme)
finaldata['CarrierHash'] = finaldata['Carrier'].str.encode("UTF-8").apply(hashme)

# remove string fields
finaldata = finaldata.drop(['OriginAirportCode', 'DestAirportCode', 'Carrier'], axis=1)

# Train model
y = finaldata['DepDel15']
x = finaldata.drop(['DepDel15'], axis=1)
x.fillna(0.005, inplace=True)
y.fillna(0, inplace=True)

for col in ['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepHour', 'OrigAirportHash', 'DestAirportHash', 'CarrierHash']:
    x[col] = x[col].astype('category')

model = RandomForestClassifier(n_estimators=10)
model.fit(x, y)

# Save model to Blob storage specified in Dockerfile
import pickle
blob_account_name = os.environ.get('ds_blob_account')
blob_account_key = os.environ.get('ds_blob_key')
mycontainer = os.environ.get('ds_container')

filename = os.environ.get('ds_model_filename')
dirname = os.getcwd()
localfile = os.path.join(dirname, filename)

# save model locally
pickle.dump(model, open(localfile, 'wb'), protocol=2)

# upload model to blob
blob_service=BlockBlobService(account_name=blob_account_name, account_key=blob_account_key)
blob_service.create_blob_from_path(mycontainer, filename, localfile)
