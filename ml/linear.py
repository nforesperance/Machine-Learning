import pandas as pd
import quandl,math
from datetime import datetime
import numpy as np
from sklearn import preprocessing,model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import pickle

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

# Note: This section was used to create pickle and commented out
# so as to avoid training the model every time

# reg = LinearRegression(n_jobs=-1)
# reg.fit(X_train,y_train)
# with open ('linearregression.pickle','wb') as f:
#     pickle.dump(reg,f)

pickle_in = open("linearregression.pickle",'rb')
reg = pickle.load(pickle_in)

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

fig, ax = plt.subplots()
ax.set_aspect('auto')
fig.autofmt_xdate()
x = pd.to_datetime(df['Date'], format='%Y-%m-%d') 


# Usual Dataset
x_old=x[:-forecast_out]
y =[d for d in df['label']]
y= y[:-forecast_out]
data = pd.DataFrame({'x':x_old, 'y':y})
datemin = np.datetime64(str((data['x'])[0].date()))

# Predicted 
y_pred = df['Forecast']
x_latest=x[-forecast_out:]
y_pred =y_pred[-forecast_out:]
pred_df = pd.DataFrame({'x':x_latest, 'y':y_pred})
last = len(pred_df['x'])-1
datemax = np.datetime64(str((pred_df['x'])[last].date()))

ax.set_xlim(datemin, datemax)
ax.set_ylim(min(data['y'])-0.2,max(pred_df['y'])+0.2)
plt.plot(data['x'],data['y'], color="red")
plt.plot(pred_df['x'],pred_df['y'], color="blue")
plt.show()
