# print('hello world')

# --------------------------------------
# Import relevant libraries
# --------------------------------------

import pandas as pd
import numpy as np
import time
import re

import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from datetime import datetime, timedelta
from pytz import timezone

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
plt.rcParams["figure.figsize"] = (12, 6)  # Change matplotlib Box Size
plt.rcParams["font.size"] = 20  # Change matplotlib Font Size

# --------------------------------------
# Load the Dataset
# --------------------------------------
folder_name = '../Data/processed/'
# name = 'device01_2020-02-08_cps_0.005_fo_3_12td'
# name = 'device35_2019-04-13_cps_0.01_fo_3_08td'
name = 'device34_2019-03-24_cps_0.005_fo_3_12td'

file_name = folder_name + name + '.csv'
df_pred = pd.read_csv(file_name, delimiter=',')

# --------------------------------------
# Convert ts_date into a datetime and convert UTC into Swiss Time
# --------------------------------------
utc_time = pd.to_datetime(df_pred['ds'], format='%Y-%m-%d %H:%M:%S', utc=True)
df_pred['local_time_to_drop'] = utc_time.apply(
    lambda x: x.tz_convert('Europe/Zurich'))
df_pred['ds'] = df_pred['local_time_to_drop']

# Drop unnecessary columns
df_pred = df_pred.drop(['Unnamed: 0', 'local_time_to_drop'], axis=1)

# --------------------------------------
# Show the data
# --------------------------------------
print('Print graph')
fig, ax = plt.subplots(figsize=(8, 4))
plt.plot('ds', 'actual', data=df_pred, label="Actual", color='black',
         linewidth=1)
plt.plot('ds', 'preds', data=df_pred, label="Prediction", color='blue',
         marker='o',
         linestyle='dashed', linewidth=0.5, markersize=2)
plt.fill_between('ds', 'lower_y', 'upper_y', data=df_pred, facecolor='b',
                 alpha=0.5)
myFmt = DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(myFmt)
plt.xlabel('Time', fontsize=10)
plt.ylabel('Parameter value', fontsize=10)
plt.title(name, fontsize=10)
plt.legend(loc='upper right')
plt.gcf().autofmt_xdate()
plt.show()
