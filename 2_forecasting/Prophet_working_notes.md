# Prophet forecasting. Work notes


### 2020 04 14:
**Test 1: Testing different sampling periods, changepoint_prior_scale and daily_fo of the Prophet model**
prophet_grid = {'df_dev' : [df_dev],
                'device' : [device],
                'parameter' : ['co2'],
                'begin' : ['2019-03-22', '2019-03-24'],
                'end' : ['2019-04-03'],
                'sampling_period_min' : [1, 5, 30],
                'graph' : [1],
                'predict_day' : [1],
                'interval_width' : [0.6],
                'changepoint_prior_scale' : list((0.01, 15)), # list(np.arange(0.01,30,1).tolist()),
                'daily_fo' : [3, 9, 18],
#                 'holidays_prior_scale' : list((1000,100,10,1,0.1)),
               }
36 models (98min)
10 first:
		device	par begin		end			SP	IW	fo	CPS     mape_average
11	[device33]	co2	2019-03-22	2019-04-03	30	0.6	3	15.00	0.116839
10	[device33]	co2	2019-03-22	2019-04-03	5	0.6	3	15.00	0.122786
9	[device33]	co2	2019-03-22	2019-04-03	1	0.6	3	15.00	0.123446
0	[device33]	co2	2019-03-22	2019-04-03	1	0.6	3	0.01	0.123622
17	[device33]	co2	2019-03-22	2019-04-03	30	0.6	18	15.00	0.128037
29	[device33]	co2	2019-03-24	2019-04-03	30	0.6	3	15.00	0.128436
14	[device33]	co2	2019-03-22	2019-04-03	30	0.6	9	15.00	0.128777
15	[device33]	co2	2019-03-22	2019-04-03	1	0.6	18	15.00	0.129947
16	[device33]	co2	2019-03-22	2019-04-03	5	0.6	18	15.00	0.130079

10 last:

22	[device33]	co2	2019-03-24	2019-04-03	5	0.6	9	0.01	0.174364
25	[device33]	co2	2019-03-24	2019-04-03	5	0.6	18	0.01	0.174383
19	[device33]	co2	2019-03-24	2019-04-03	5	0.6	3	0.01	0.176853
2	[device33]	co2	2019-03-22	2019-04-03	30	0.6	3	0.01	0.189457
5	[device33]	co2	2019-03-22	2019-04-03	30	0.6	9	0.01	0.201878
8	[device33]	co2	2019-03-22	2019-04-03	30	0.6	18	0.01	0.202137
20	[device33]	co2	2019-03-24	2019-04-03	30	0.6	3	0.01	0.283187
26	[device33]	co2	2019-03-24	2019-04-03	30	0.6	18	0.01	0.284897
23	[device33]	co2	2019-03-24	2019-04-03	30	0.6	9	0.01	0.286086

Next:
'changepoint_prior_scale' (CPS)
	- greater than 1 just make a aberrantly wide prediction: 15 IS NOT GOOD !
	- Higher than 1 significantly decreases the accuracy!
	- try changepoint_prior_scale : 0.001, 0.01, 0.1

'sampling_period_min' : [1, 5, 30]:
	- you miss event with 30min, but are faster
	- try 1, 2 and 5

'begin':
	- 10 days is better than 6 days but much longer analysis
	- Keep 6 days and may be for the final round increase the number of days

'daily_fo' : [3, 9, 18]
	- Lower is definitely better than greater! 
	- try 2, 3, 6

#########################
**Test 2: Testing different shorter sampling periods, changepoint_prior_scale and daily_fo of the Prophet model**
prophet_grid = {'df_dev' : [df_dev],
                'device' : [device],
                'parameter' : ['co2'],
                'begin' : ['2019-03-24'],
                'end' : ['2019-04-03'],
                'sampling_period_min' : [1, 2, 5],
                'graph' : [1],
                'predict_day' : [1],
                'interval_width' : [0.6],
                'changepoint_prior_scale' : [0.001, 0.01, 0.1], # list(np.arange(0.01,30,1).tolist()),
                'daily_fo' : [2, 3, 6],
#                 'holidays_prior_scale' : list((1000,100,10,1,0.1)),
               }
top ten:
		device	par begin		end			SP	IW	fo	CPS     mape_average
18	[device33]	co2	2019-03-24	2019-04-03	1	0.6	2	0.100	0.135834
24	[device33]	co2	2019-03-24	2019-04-03	1	0.6	6	0.100	0.136738
21	[device33]	co2	2019-03-24	2019-04-03	1	0.6	3	0.100	0.138159
19	[device33]	co2	2019-03-24	2019-04-03	2	0.6	2	0.100	0.141241
25	[device33]	co2	2019-03-24	2019-04-03	2	0.6	6	0.100	0.142547
22	[device33]	co2	2019-03-24	2019-04-03	2	0.6	3	0.100	0.143527
23	[device33]	co2	2019-03-24	2019-04-03	5	0.6	3	0.100	0.148306
26	[device33]	co2	2019-03-24	2019-04-03	5	0.6	6	0.100	0.148327

worst 5
7	[device33]	co2	2019-03-24	2019-04-03	2	0.6	6	0.001	0.241893
4	[device33]	co2	2019-03-24	2019-04-03	2	0.6	3	0.001	0.243747
2	[device33]	co2	2019-03-24	2019-04-03	5	0.6	2	0.001	0.250573
5	[device33]	co2	2019-03-24	2019-04-03	5	0.6	3	0.001	0.259491
8	[device33]	co2	2019-03-24	2019-04-03	5	0.6	6	0.001	0.280115

'sampling_period_min' : [1, 2, 5]:
	- the smaller is the better: use 1

'changepoint_prior_scale' : [0.001, 0.01, 0.1]
	- 0.1 is too large ! The prediction goes wide and the prediction is meaningless.
	- 0.001 seems to give visually better results than 0.01 even if the median mape is lower for 0.01 than 0.01
	- Keep both 0.01 and 0.001

'daily_fo' : [2, 3, 6]:
	- Would choose 3. The dynamic is not only binary. choose 3 or 6. 6.

#########################
**Test 3: Testing different changepoint_prior_scale and daily_fo of the Prophet model**
prophet_grid = {'df_dev' : [df_dev],
                'device' : [device],
                'parameter' : ['co2'],
                'begin' : ['2019-03-24', '2019-03-26', '2019-03-28'],
                'end' : ['2019-04-03'],
                'sampling_period_min' : [1],
                'graph' : [1],
                'predict_day' : [1],
                'interval_width' : [0.6],
                'changepoint_prior_scale' : [0.001, 0.01], # list(np.arange(0.01,30,1).tolist()),
                'daily_fo' : [3, 6, 9],
#                 'holidays_prior_scale' : list((1000,100,10,1,0.1)),
               }
top 6:
		device	par begin		end			SP	IW	fo	CPS     mape_average
3	[device33]	co2	2019-03-22	2019-04-03	1	0.6	3	0.010	0.123622
4	[device33]	co2	2019-03-22	2019-04-03	1	0.6	6	0.010	0.130818
5	[device33]	co2	2019-03-22	2019-04-03	1	0.6	9	0.010	0.131206
9	[device33]	co2	2019-03-24	2019-04-03	1	0.6	3	0.010	0.155955
10	[device33]	co2	2019-03-24	2019-04-03	1	0.6	6	0.010	0.157499
11	[device33]	co2	2019-03-24	2019-04-03	1	0.6	9	0.010	0.159407

'changepoint_prior_scale' : [0.001, 0.01, 0.1]
	- 0.1 is too large ! The prediction goes wide and the prediction is meaningless.
	- 0.01 may be better than 0.001
	- Keep both 0.01 and 0.001

'begin' : ['2019-03-22', '2019-03-24', '2019-03-26']
	- In this case, the more data is used for the model fitting, the lower is the median_mape
	- In other, using taking 

#########################
**Test 4: Testing different predicted days, training periods, daily_fo of the Prophet model**

		device	par begin		end			SP	IW	fo	CPS     mape_average
0	[device33]	co2	2019-03-22	2019-04-03	1	0.6	3	0.01	0.123622
1	[device33]	co2	2019-03-22	2019-04-03	1	0.6	6	0.01	0.130818
2	[device33]	co2	2019-03-22	2019-04-03	1	0.6	9	0.01	0.131206
3	[device33]	co2	2019-03-24	2019-04-03	1	0.6	3	0.01	0.155955
4	[device33]	co2	2019-03-24	2019-04-03	1	0.6	6	0.01	0.157499
5	[device33]	co2	2019-03-24	2019-04-03	1	0.6	9	0.01	0.159407

0	[device33]	co2	2019-03-21	2019-04-02	1	0.6	3	0.01	0.112987
2	[device33]	co2	2019-03-21	2019-04-02	1	0.6	9	0.01	0.118718
1	[device33]	co2	2019-03-21	2019-04-02	1	0.6	6	0.01	0.119256
3	[device33]	co2	2019-03-23	2019-04-02	1	0.6	3	0.01	0.122369
5	[device33]	co2	2019-03-23	2019-04-02	1	0.6	9	0.01	0.132954
4	[device33]	co2	2019-03-23	2019-04-02	1	0.6	6	0.01	0.136060

3	[device33]	co2	2019-03-22	2019-04-01	1	0.6	3	0.01	0.106763
5	[device33]	co2	2019-03-22	2019-04-01	1	0.6	9	0.01	0.122046
4	[device33]	co2	2019-03-22	2019-04-01	1	0.6	6	0.01	0.124156
2	[device33]	co2	2019-03-20	2019-04-01	1	0.6	9	0.01	0.269513
0	[device33]	co2	2019-03-20	2019-04-01	1	0.6	3	0.01	0.271532
1	[device33]	co2	2019-03-20	2019-04-01	1	0.6	6	0.01	0.273855

3	[device33]	co2	2019-03-21	2019-03-31	1	0.6	3	0.01	0.095746
5	[device33]	co2	2019-03-21	2019-03-31	1	0.6	9	0.01	0.099277
4	[device33]	co2	2019-03-21	2019-03-31	1	0.6	6	0.01	0.100746
1	[device33]	co2	2019-03-19	2019-03-31	1	0.6	6	0.01	0.387377
0	[device33]	co2	2019-03-19	2019-03-31	1	0.6	3	0.01	0.390950
2	[device33]	co2	2019-03-19	2019-03-31	1	0.6	9	0.01	0.391391

5	[device33]	co2	2019-03-20	2019-03-30	1	0.6	9	0.01	0.298872
4	[device33]	co2	2019-03-20	2019-03-30	1	0.6	6	0.01	0.300329
3	[device33]	co2	2019-03-20	2019-03-30	1	0.6	3	0.01	0.304640
2	[device33]	co2	2019-03-18	2019-03-30	1	0.6	9	0.01	0.388522
1	[device33]	co2	2019-03-18	2019-03-30	1	0.6	6	0.01	0.390094
0	[device33]	co2	2019-03-18	2019-03-30	1	0.6	3	0.01	0.393925

The fo of 3 is most often the best
But the prediction is not necessarily better with more days (compare prediction of 04 01 and 03 31) --> then it make sense to do a grid search with increasing number of days.

### 2020 04 14:

* Let Prophet handle alone the NA.
* discard aberrant values
* Check random search to see the best informative model: https://github.com/facebook/prophet/issues/549

	initial	horizon	period	    mse	            rmse	    mae	       mape	    coverage
0	8 days	1 days	1 days	9154.673866	    95.680060	82.031186	0.133898	0.181250
0	1 days	1 days	1 days	12075.441899	109.888316	88.027051	0.140549	0.191176
0	4 days	1 days	1 days	11944.199185	109.289520	89.475573	0.143288	0.176820
0	4 days	1 days	8 days	12897.974381	113.569249	100.888793	0.157171	0.072165
0	8 days	1 days	8 days	12897.974381	113.569249	100.888793	0.157171	0.061856
0	8 days	1 days	4 days	13008.907075	114.056596	102.029925	0.159642	0.062500
0	4 days	1 days	4 days	16377.037454	127.972800	107.015331	0.160635	0.077720
0	1 days	1 days	4 days	18614.378047	136.434519	113.694468	0.174221	0.091667
0	1 days	1 days	8 days	17762.402900	133.275665	114.020323	0.180944	0.069444

0	8 days	1 days	0.5 d 	8382.943514		91.558416	77.282403	0.125738	0.209375
0	4 days	1 days	0.5 d 	10523.978522	102.586444	83.602011	0.133608	0.190335
0	8 days	1 days	1 days	9154.673866	    95.680060	82.031186	0.133898	0.177083
0	1 days	1 days	0.5 d 	11872.129916	108.959304	86.588446	0.137601	0.199142
0	1 days	1 days	1 days	12075.441899	109.888316	88.027051	0.140549	0.196078
0	4 days	1 days	1 days	11944.199185	109.289520	89.475573	0.143288	0.187221

0	12 days	1 days	0.5 d	7943.206990		89.124671	76.908726	0.126626	0.197917
0	3 days	1 days	0.5 d	10206.561500	101.027528	81.813362	0.130603	0.202224
0	6 days	1 days	0.5 d	9741.611008		98.699600	81.724256	0.131423	0.200521


																		   mape     coverage
Trained on the data from the 2019-03-14 to the 2019-03-28 (14 days)		0.137317	0.197917
Trained on the data from the 2019-03-16 to the 2019-03-28 (12 days)		0.14431	    0.197917
Trained on the data from the 2019-03-21 to the 2019-03-28 (6 days)      0.169342	0.166667

Trained on the data from the 2019-03-18 to the 2019-03-28 (10 days) 	0.174096	0.141369

Trained on the data from the 2019-03-15 to the 2019-03-27 (12 days)		0.135401	0.21875
model.add_seasonality(name='daily', period=1, fourier_order=12) # prior scale
model.add_seasonality(name='half_day', period=0.5, fourier_order=10)

Trained on the data from the 2019-03-15 to the 2019-03-27 (12 days)		0.151676	0.1812
model.add_seasonality(name='weekly', period=7, fourier_order=5) 
model.add_seasonality(name='daily', period=1, fourier_order=12)

Trained on the data from the 2019-03-15 to the 2019-03-27 (12 days)		0.15154     0.177083
model.add_seasonality(name='weekly', period=7, fourier_order=5)
model.add_seasonality(name='daily', period=1, fourier_order=12)
model.add_seasonality(name='half_day', period=0.5, fourier_order=10)


Changepoint_prior_scale --> Do not put more than 0.1
ideal is <0.01 = **Example: changepoint_prior_scale=0.005**


### 2020 04 13:

* Set up the performance metrics. --> checked it but not obvious helpful metrics is provided
* Normalize the max value (in co2 and light for instance) to facilitate the detection.
* Implement time management in df_generator (if begin_str=None, begin_str='0:00') --> OK
**Note: Too many days screws the prediction.**
**Increased the fourier_order for a better fitting and reduce time-frame was improving the fitting**

### 2020 04 11:

* Plot only the predicted dataframe for outlier interpretation --> OK
* Dynamic plot: possibility to zoom --> OK
* Set up the performance metrics.
* Normalize the max value (in co2 and light for instance) to facilitate the detection.
* Implement time management in df_generator (if begin_str=None, begin_str='0:00')


### 2020 04 09:

* set up data fetching from aws s3 bucket --> OK
* Increase processing speed of df using df_dev (device specific df) --> OK

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
* `co2`: regular. daily pattern possible to visualy identify.
* `light`: regular. daily pattern possible to visualy identify untill the 05/08 but afterward very small (?)
* `temperature`: daily pattern possible to visualy identify. General trend may prevent for pattern detection: The higher value from 24/07 to 04/08 is lower than the lower value from 06/08 to 11/08.
* `humidity`: irregular. No daily pattern possible to visualy identify. Making sense out of this data may not be worth, considering the relevance of co2.
