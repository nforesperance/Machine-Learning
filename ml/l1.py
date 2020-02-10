import pandas as pd
import quandl,math
from datetime import datetime
import numpy as np
from sklearn import preprocessing,model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates

style.use('ggplot')

# Remote Dataset (limited 50 calls per day)
# df = quandl.get('WIKI/GOOGL')
# df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
# df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Close']*100.0
# df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0

# df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
# forecast_col = 'Adj. Close'

#Local Dataset 

df = pd.read_table("aadr.us.txt",header=0,sep =",")
df['HL_PCT'] = (df['High']-df['Low'])/df['Close']*100.0
df['PCT_change'] = (df['Close']-df['Open'])/df['Open']*100.0


df = df[['Date','Close','HL_PCT','PCT_change','Volume']]
forecast_col = 'Close'

df.fillna(-9999, inplace=True)

#Number of days to predict
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
#After the shift, the forecast_out last elements of the label are NaN
#Hence we drop them
df.dropna(inplace=True)



#NB
# -our X is compost of the stack now and the various variations
# -our y is the closing stock 'forecast_out' into the future
X = np.array(df.drop(['label','Date'],1))
X = preprocessing.scale(X)
y = np.array(df['label'])[:-forecast_out]
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

reg = LinearRegression()
reg.fit(X_train,y_train)
accuracy = reg.score(X_test,y_test)

forecast_set = reg.predict(X_lately)

df['Forecast'] = np.nan
#Arranging dates for the predicted values
last_date = df.iloc[-1].Date
date_object = datetime.strptime(last_date, '%Y-%m-%d')
last_unix = datetime.timestamp(date_object)
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.fromtimestamp(next_unix).date()
    next_unix += 86400
    df.loc[next_date] = [next_date]+[np.nan for _ in range(len(df.columns)-2)]+[i]


# print(df.tail(16))
x = pd.to_datetime(df['Date'], format='%Y-%m-%d') 
y = [d for d in df['Forecast']]
print(x)


# print(x[-forecast_out:])

# plt.plot(df['Date'],df['Close'], label = "Close", color="blue")
# plt.plot((df['year'])[-forecast_out:],(df['Forecast'])[-forecast_out:], label = "Forecast", color="red")
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()
x_latest=np.c_[x[-forecast_out:]]
x_l=x[-forecast_out:]
# print(x_l)
y_latest =np.c_[y[-forecast_out:]]
y_l =y[-forecast_out:]
# print(y_l)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.gca().xaxis.set_xlim([datetime.date(2010, 10, 20), datetime.date(2010, 11, 4)])
# plt.plot(x_latest,y_latest)
# plt.gcf().autofmt_xdate()
# plt.show()

data = pd.DataFrame({'x':x_l, 'y':y_l})
# print(data.head())
fig, ax = plt.subplots()
ax.set_aspect('auto')
fig.autofmt_xdate()
datemin = np.datetime64('2017-10-20')
datemax = np.datetime64('2017-11-04')
ax.set_xlim(datemin, datemax)
ax.set_ylim(55,60)
plt.plot(data['x'],data['y'], label = "Close", color="blue")
plt.show()
