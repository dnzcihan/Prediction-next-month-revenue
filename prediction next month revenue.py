
# coding: utf-8

# In[205]:


import pandas as pd
import matplotlib as plt
import fbprophet

class Pre:
    df1=pd.read_csv('users.csv')
    df2= pd.read_csv('voices.csv')
    
    def prediction(self,df1,df2):
        
        self.df1 = df1
        self.df2 = df2
        
        e =df2.groupby(['date','userId'])['amount'].mean()
        e = e.to_frame()
        e.to_csv('a.csv')
        newDf = pd.read_csv('a.csv')
        newDf['ds'] =pd.to_datetime(newDf['date'], format='%Y.%m.%d')
        newDf['y'] = newDf['amount']
        newDf.drop(['date','amount'],axis=1)
        df_prophet = fbprophet.Prophet()
        df_prophet.fit(newDf)
 
        # 1 aylık gelecek tahmini yap
        tahmin_suresi=30
        df_forecast = df_prophet.make_future_dataframe(periods= tahmin_suresi, freq='D')
 
        # Tahminleri gerçekleştir
        df_forecast = df_prophet.predict(df_forecast)


        df_prophet.plot(df_forecast, xlabel = 'Tarih', ylabel = 'amount')

        return  print('30  gün sonra  toplam ödeme : ',df_forecast[-30:]['yhat'].sum())




tahmin=Pre()

tahmin.prediction(df1=pd.read_csv('users.csv'),df2=pd.read_csv('voices.csv'))

