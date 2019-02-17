# Time Series

import pandas as pd
import numpy as np

t1 = pd.Timestamp('2013-01-01')
t2 = pd.Timestamp('2013-01-01 21:15:06')
t3 = pd.Timestamp('2013-01-01 21:15:06.7')
p  = pd.Period('2013-01-01',freq = 'M')

dr1 = pd.date_range('2013-01-01','2013-12-31',freq = 'D')
dr2 = pd.date_range('2013-01-01','2013-12-31',freq = 'M')
dr3 = pd.date_range('2013-01-01','2013-12-31',freq = 'M')

# date as index

dr = pd.date_range('2013-01-01','2013-12-31',freq='M')
df = pd.DataFrame(np.arange(12),dr,columns=['Qty'])
print(dr)
print(df)
dr3-dr2

import datetime
print("Current date and time:", datetime.datetime.now())
print("Current Year:",datetime.date.today().strftime("%Y"))
print("Month of Year:",datetime.date.today().strftime("%B"))
print("Week number of the Year:",datetime.date.today().strftime("%W"))
print("Weekday of the Week:",datetime.date.today().strftime("%w"))

import seaborn as sns
import matplotlib.pyplot as plt
planets = sns.load_dataset('planets')
sns.factorplot('year',data = planets,kind = 'count')
sns.factorplot('year',data = planets, hue = 'method', kind = 'count', aspect =4)
plt.ylabel('No of planets discovered')
plt.xlabel('Year')

import quandl
quandl.ApiConfig.api_key = ''


tesla = quandl.get('WIKI/TSLA')
gm = quandl.get('WIKI/GM')
tesla.head()
gm.head()

plt.plot(tesla.index,tesla['Close'],label = 'TESLA')
plt.plot(gm.index,gm['Close'],label= 'GM')
plt.legend()
plt.title('Stock Price')

