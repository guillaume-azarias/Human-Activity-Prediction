# Prophet forecasting

### 2020 04 08:

* What are the black dots ? *df time point*

* Where is the plotting function ? *in prophet_fit*

* Id df really resampled ? *Yes*

* Understand seasonality ?
	*period is in day*

* Adapt seasonality to acquisition frequency to match the number of points per night and day, start df on a day: *OK*

* feature engineering with humidity and temp? *Considering the variability in humidity, it might be better to leave this parameter out of the data to analyse*

check if others have used prophet for lower time scale

Possibility of analysis pipeline

    1) Identify day and night using Prophet --> OK
    
    2) Identify outlier
    
    3) Identify when outlier phase starts and ends
    
    4) Cluster outlier activities defined by amplitude/area and duration


**Model parameter analysis**

* interval_width=0.9, *anomaly threshold*
* yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False *--> Use your own seasonility*
* changepoint_prior_scale=0.01) # Adjusting trend flexibility. low *--> toward overfit. Does not make sense to have more than 0.1. Just increasing tolerance*

* model.add_seasonality(name='daily', period=1,
	fourier_order=2
	prior_scale=0.1

* model.add_seasonality(name='half_day', period=0.5
	fourier_order=10
	prior_scale=?

* today_index = *index. Take the binning in account for calculation. Current binning is 30min*

* lookback_n = int(today_index*0.9) *I choose to take 90% of the df*

* predict_n = 144 *in lines / data points. Take the binning in account for calculation*

**Data type analysis**

*device31: co2 from the 2019-07-13 until the 2019-08-11:*
* co2: regular. daily pattern possible to visualy identify.
* light: regular. daily pattern possible to visualy identify untill the 05/08 but afterward very small (?)
* temperature: daily pattern possible to visualy identify. General trend
* co2: regular. daily pattern possible to identify.
### 2020 04 09:

* Plot only the predicted dataframe