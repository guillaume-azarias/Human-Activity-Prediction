"""
This script centralized the functions used in the notebook Prophet_Prediction

The functions in this script are:
* load_ds: Load a device-specific dataframe located in the folder ../data/interim.
  If it does not exist, it will start load_raw_data that will create all the
  device-specific dataframes.
* df_dev_formater: Load a csv file containing the data from a single device
  and return a pandas dataframe where the date and time column is formatted
  in datetime for Switzerland.
* find_index: Find the index of a date and time for a specific dataframe.
  Can be helpful for data exploration.
* df_generator: Prepare the dataframe according to be processed by Prophet.
  Critical function of Prophet_Prediction: please read the description within the
  function definition.
* prophet_fit: Fit the model to the time-series data and generate forecast
  for specified time frames. Version of the tutorial modified to work with
  shorter time frame than entire days (main limitation of the tutorial).
* prophet_plot: Plot actual, predictions, and anomalous values. Writing of
  the label for anomalies were disabled. Version of the tutorial modified to
  work with shorter time frame than entire days.
* get_outliers: Combine the actual values and forecast in a data frame and
  identify the outliers. From the tutorial.
* execute_cross_validation_and_performance_loop: Execute Cross Validation
  and Performance Loop. This function is taken from this article
  (https://medium.com/@jeanphilippemallette/prophet-auto-selection-with-cross-validation-7ba2c0a3beef) from Jean-Philippe Mallette.
* prophet: Run a single instance of prophet with the relevant parmeters
* GridSearch_Prophet: GridSearch tool for hyperparameter tuning of prophet

---------------------------------------------------------------------------
Note: This script should be located in the same folder as Prophet_Prediction.ipynb
---------------------------------------------------------------------------

This script was PEP8 formatted using autopep8. To do it, enter this line in
the Terminal:

autopep8 --in-place --aggressive --aggressive <filename>

"""

from pytz import timezone
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet import Prophet
import fbprophet
import pandas as pd
import time
import re

import seaborn as sns
sns.set()


def load_ds(device_nb):
    '''
    This is a script to load a device-specific dataset.
    
    Arg:
        - device_nb: String. 2-digit number, as a string of the device you want to look at
    
    Return
        - device: List. List of device(s) containing in the dataframe. Normally, it is one device but it is checked afterwards. 
        - df_dev: Dataframe containing the data of a specific device. To pass to df_generator
    '''
    try:
        device, df_dev = df_dev_formater(device_nb)

        assert device.shape[0] == 1, 'No, or multiple devices in the df'

        # Check report:
        print('Check report:\n##############################################')
        print('Device contained in the dataset: ' + device)
        print('Tenant using the device: ' + df_dev['tenant'].unique())
        # print('\nThere are ' + str(df_dev.shape[0]) + ' lines.')
        print('\nData types:')
        print(df_dev.dtypes)
        last = df_dev.shape[0] - 1
        print('\nAvailable data from the {:%Y-%m-%d} to the {:%Y-%m-%d}.'.format(
            df_dev['ds'][0], df_dev['ds'][last]))

        return device, df_dev
    except:
        print('Device-specific dataframe not found in the folder ../data/interim')
        # Start a script to load the raw data
        load_raw_data()


def df_dev_formater(device_nb):
    """
    This is a script to load a csv file created by df_dev_generator.
    It loads a csv file, drop the 'Unnamed: 0' column and convert the
    ts_date column (object) into a column named ds and formated in
    local Swiss time (datetime).

    Arg:
        - device_nb: String. 2 digit number of the device

    Returns:
        - df_dev: dataframe to pass to df_generator
    """
    # Data Loading
    file_name = '../Data/interim/device' + str(device_nb) + '_alldays.csv'

    df_raw = pd.read_csv(file_name, delimiter=',')

    # Clean the dataset
    df_raw = df_raw.drop(['Unnamed: 0'], axis=1)

    # Convert ts_date into a datetime and convert UTC into Swiss Time
    utc_time = pd.to_datetime(
        df_raw['ts_date'], format='%Y-%m-%d %H:%M:%S', utc=True)
    df_raw['local_time_to_drop'] = utc_time.apply(
        lambda x: x.tz_convert('Europe/Zurich'))

    # Drop unnecessary columns
    df_raw['ts_date'] = df_raw['local_time_to_drop']
    df_raw.rename({'ts_date': 'ds'}, axis=1, inplace=True)
    df_dev = df_raw.drop(['local_time_to_drop'], axis=1)

    # Make sure that only the relevant device is included in df_dev
    device = df_dev['device'].unique()

    return device, df_dev


def find_index(dataframe, date, starting_time=None):
    """
    This function allows to find the index for a specific dataframe, date and time.

    Args:
        - dataframe: pandas dataframe of interest
        - date: string formatted as '2019-04-08'
        - time: string formatted as '20:30'. If None, time = '0:00'.

    Print the index
    """
    assert dataframe.shape[0] > 0, 'The dataframe is empty !'

    if not starting_time:
        # day time of the first day of the df. Might be relevant to get a full
        # day and help the day/night clustering
        starting_time = '0:00'

    date_str = date + ' ' + starting_time
    date_dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M')

    # Apply the Swiss time zone.
    # http://pytz.sourceforge.net/#localized-times-and-date-arithmetic
    swiss = timezone('Europe/Zurich')
    date_dt = swiss.localize(date_dt)

    # Filter according to begin and end. Does not take in account the starting
    # time...
    date_on = dataframe[(dataframe['ds'] >= date_dt)]
    index = date_on.shape[0]

    print(index)


def df_generator(df_dev, device, parameter, begin, end, sampling_period_st,
                 sampling_period_num, graph=None, predict_day=1):
    """
    This function is generating a new dataframe from entire dataframe.
    Note that for now, df_dev is the device 31 specific dataframe.

    Args:
        - df_dev: Dataframe. Full dataframe with
            o device                                object
            o tenant                                object
            o ds             datetime64[ns, Europe/Zurich]
            o light                                float64
            o temperature                          float64
            o humidity                             float64
            o co2                                  float64
        - parameter: String. among 'light', 'temperature', 'humidity', 'co2'.
          co2 might be the more "human-activity" related
        - begin: String. Day of the beginning of the new dataframe.
        - end: String. Day of the end of the new dataframe.
        - sampling_period_st: String. Duration of bin for data downsampling. !
          Format is not accurate for date calculations.
        - sampling_period_num: Float. Number of hours of the sampling_period_st.
          Example: resampling every 30min: '0.5
        - graph=None: Set to None to show the graph and a value if you don't want
          to show the graph.
        - predict_day=1. Number of days predicted. 1 by default.

    Returns:
        df: pandas dataframe for the specific parameter
        predict_n = Int. Number of data points to predict.
        today_index = df.shape[0] - predict_n # index
        lookback_n = int(today_index*0.99)

    Note: Real values of predict_n, today_index and lookback_n depend on
          sampling_period_st and sampling_period_num. Wrong indications of
          sampling_period_st or sampling_period_num can lead to wrong predictions.

    TODO: Check if any existing function converts sampling_period_st into
          sampling_period_st and vice-versa. Use a Regex-based function could take
          care of it and avoid miscalculations.
    """

    # Saving name
    name = str(device) + ': ' + str(parameter) + ' from the ' + \
        str(begin) + ' until the ' + str(end)

    # Prepare the dates
    # day time of the first day of the df. Might be relevant to get a full day
    # and help the day/night clustering
    starting_time = '21:00'

    begin_str = begin + ' ' + starting_time
    end_str = end + ' ' + starting_time

    begin_dt = datetime.strptime(begin_str, '%Y-%m-%d %H:%M')
    end_dt = datetime.strptime(end_str, '%Y-%m-%d %H:%M')

    # Apply the Swiss time zone.
    # http://pytz.sourceforge.net/#localized-times-and-date-arithmetic
    swiss = timezone('Europe/Zurich')
    # swiss.zone
    begin_dt = swiss.localize(begin_dt)
    end_dt = swiss.localize(end_dt)

    # Sorry this is not elegant. Fix it
    pd.options.mode.chained_assignment = None  # default='warn'

    # Filter according to begin and end.
    df_full = df_dev[(df_dev['ds'] >= begin_dt) & (df_dev['ds'] <= end_dt)]

    # Generate a parameter-specific df
    df_full.rename(columns={parameter: 'y'}, inplace=True)
    df_parameter = df_full[['ds', 'y']]

    # Set ds as the index
    df_parameter.index = df_parameter.ds
    df_parameter.reindex
    df_original = df_parameter.copy()

    # Downsampling and fill NaN. Need to have set ds as the index.
    # TODO: Disable the pad function to let Prophet deal with missing data
    df = df_parameter.resample(sampling_period_st).pad()
    df = df.iloc[1:]

    if not graph:
        # Plot the df
        fig_df, ax = plt.subplots(figsize=(8, 4))
        # Data with original frequency
        df_original.y.plot(label="Original", color='gray', linewidth=1)
        df.y.plot(label="Resampled data", color='black', marker='o',
                  linestyle='dashed', linewidth=0.5, markersize=2)

        myFmt = DateFormatter("%d/%m %H:%M")
        ax.xaxis.set_major_formatter(myFmt)
        plt.xlabel('Time', fontsize=10)
        plt.ylabel(parameter, fontsize=10)
        plt.title(name, fontsize=10)
        plt.legend(loc='upper right')

        # vertical lines
        begin_str_vl = begin + ' 0:00'
        end_str_vl = end + ' 0:00'

        begin_dt_vl = datetime.strptime(
            begin_str_vl, '%Y-%m-%d %H:%M') + timedelta(days=1)
        end_dt_vl = datetime.strptime(end_str_vl, '%Y-%m-%d %H:%M')

        begin_dt_vl = swiss.localize(begin_dt_vl)
        end_dt_vl = swiss.localize(end_dt_vl)

        daterange = pd.date_range(begin_dt_vl, end_dt_vl)
        for single_date in daterange:
            plt.axvline(x=single_date, color='lightseagreen', linestyle='--')
        # plt.show()

#         # If you want to save the file
#         folder = '/Users/path_to_your_folder/figures/'
#         filename = folder + name + '.png'
#         plt.savefig(filename, bbox_inches = "tight")
#     else:
#         None

    # Shape report
    last_df = df.shape[0] - 1
    last = df_dev.shape[0] - 1
    print('Full dataset: {:%Y-%m-%d} to the {:%Y-%m-%d}. Analysed data the {:%Y-%m-%d} to the {:%Y-%m-%d}.'
          .format(df_dev['ds'][0], df_dev['ds'][last], df['ds'][0], df['ds'][last_df]))

    # specify the time frames.
    predict_n = int(predict_day * 24 / sampling_period_num)  # in data points
    today_index = df.shape[0] - predict_n  # index
    lookback_n = int(today_index * 0.99)  # 1 week 336

    return df, predict_n, today_index, lookback_n


# Code reference:
# https://github.com/paullo0106/prophet_anomaly_detection/blob/master/utils.py

def prophet_fit(
        df,
        prophet_model,
        today_index,
        sampling_period_st,
        sampling_period_num,
        lookback_days=None,
        predict_days=21):
    """
    Fit the model to the time-series data and generate forecast for specified time frames
    Args
    ----
    df : pandas DataFrame
        The daily time-series data set contains ds column for
        dates (datetime types such as datetime64[ns]) and y column for numerical values
    prophet_model : Prophet model
        Prophet model with configured parameters
    today_index : int
        The index of the date list in the df dataframe, where Day (today_index-lookback_days)th
        to Day (today_index-1)th is the time frame for training
    sampling_period_st: string. Period of resampling.
        Relevant parameter for make_future_dataframe. Example: resampling every 30min: '30T'
    sampling_period_num: float. Number of hours of the sampling_period_st. Example: resampling every 30min: '0.5
    lookback_days: int, optional (default=None)
        As described above, use all the available dates until today_index as
        training set if no value assigned
    predict_days: int, optional (default=21)
        Make prediction for Day (today_index)th to Day (today_index+predict_days)th
    Returns
    -------
    fig : matplotlib Figure
        A plot with actual data, predicted values and the interval
    forecast : pandas DataFrame
        The predicted result in a format of dataframe
    prophet_model : Prophet model
        Trained model
    """

    # segment the time frames
    lookback_days_show = int(lookback_days / (24 / sampling_period_num))
    predict_days_show = int(predict_days / (24 / sampling_period_num))

    baseline_ts = df['ds'][:today_index]
    baseline_y = df['y'][:today_index]
    if not lookback_days:
        print('o Trained on data from the {:%Y-%m-%d} to the {:%Y-%m-%d} ({} days).'.format(
            df['ds'][0], df['ds'][today_index - 1], lookback_days_show))
    else:
        baseline_ts = df['ds'][today_index - lookback_days:today_index]
        baseline_y = df.y[today_index - lookback_days:today_index]
        print('o Trained on the data from the {:%Y-%m-%d} to the {:%Y-%m-%d} ({} days).'.format(
            df['ds'][today_index - lookback_days], df['ds'][today_index - 1], lookback_days_show))
    print('o Predict from the {:%Y-%m-%d} to the {:%Y-%m-%d} ({} days).'.format(
        df['ds'][today_index], df['ds'][today_index + predict_days - 1], predict_days_show))

    # fit the model
    prophet_model.fit(pd.DataFrame({'ds': baseline_ts.values,
                                    'y': baseline_y.values}))

    # make prediction. Note that the frequency of the sampling period used for the dataframe
    # To make a new frequency, check
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
    future = prophet_model.make_future_dataframe(
        periods=predict_days, freq=sampling_period_st)
#     future = prophet_model.make_future_dataframe(periods=predict_days, freq = '30T')
    forecast = prophet_model.predict(future)

    # generate the plot
    # myFmt = DateFormatter("%d/%m %H:%M")
    # ax.xaxis.set_major_formatter(myFmt)
    fig = prophet_model.plot(forecast,
                             xlabel='Time',
                             ylabel='Parameter value',
                             figsize=(7, 5)
                             )


    return fig, forecast, prophet_model


def prophet_plot(
        df,
        fig,
        today_index,
        lookback_days=None,
        predict_days=21,
        outliers=list()):
    """
    Plot the actual, predictions, and anomalous values
    Args
    ----
    df : pandas DataFrame
        The daily time-series data set contains ds column for
        dates (datetime types such as datetime64[ns]) and y column for numerical values
    fig : matplotlib Figure
        A plot with actual data, predicted values and the interval which we previously obtained
        from Prophet's model.plot(forecast).
    today_index : int
        The index of the date list in the dataframe dividing the baseline and prediction time frames.
    lookback_days : int, optional (default=None)
        Day (today_index-lookback_days)th to Day (today_index-1)th is the baseline time frame for training.
    predict_days : int, optional (default=21)
        Make prediction for Day (today_index)th to Day (today_index+predict_days)th.
    outliers : a list of (datetime, int) tuple
        The outliers we want to highlight on the plot.
    """
    # retrieve the subplot in the generated Prophets matplotlib figure
    ax = fig.get_axes()[0]

    start = 0
#     end = today_index + predict_days # Original code
    end = df.shape[0]
    x_pydatetime = df['ds'].dt.to_pydatetime()
    # highlight the actual values of the entire time frame
    ax.plot(x_pydatetime[start:end],
            df.y[start:end],
            color='orange', label='Actual')

    # plot each outlier in red dot and annotate the date
    for outlier in outliers:
        ax.scatter(outlier[0], outlier[1], s=16, color='red', label='Anomaly')
#         ax.text(outlier[0], outlier[1], str(outlier[0])[:10], color='red', fontsize=6)

    # highlight baseline time frame with gray background
    if lookback_days:
        start = today_index - lookback_days
    ax.axvspan(x_pydatetime[start],
               x_pydatetime[today_index],
               color=sns.xkcd_rgb['grey'],
               alpha=0.2)

    # annotate the areas, and position the text at the bottom 5% by using ymin
    # + (ymax - ymin) / 20
    ymin, ymax = ax.get_ylim()[0], ax.get_ylim()[1]
    ax.text(x_pydatetime[int((start + today_index) / 2)],
            ymin + (ymax - ymin) / 20, 'Baseline area')
    ax.text(x_pydatetime[int((today_index * 2 + predict_days) / 2)],
            ymin + (ymax - ymin) / 20, 'Prediction area')

    # re-organize the legend
    patch1 = mpatches.Patch(color='red', label='Anomaly')
    patch2 = mpatches.Patch(color='orange', label='Actual')
    patch3 = mpatches.Patch(color='skyblue', label='Predict and interval')
    patch4 = mpatches.Patch(color='grey', label='Baseline area')
    plt.legend(handles=[patch1, patch2, patch3, patch4])


    # If you want to save the file
    folder = '/Users/path_to_your_folder/figures/'
    name = 'Prediction_' + df.iloc[today_index, 0].strftime('%Y-%m-%d')
    filename = folder + name + '.png'
    plt.savefig(filename, bbox_inches = "tight")
    plt.show()


def get_outliers(df, forecast, today_index, predict_days=21):
    """
    Combine the actual values and forecast in a data frame and identify the outliers

    Args
    ----
    df : pandas DataFrame
        The daily time-series data set contains ds column for
        dates (datetime types such as datetime64[ns]) and y column for numerical values
    forecast : pandas DataFrame
        The predicted result in a dataframe which was previously generated by
        Prophet's model.predict(future)
    today_index : int
        The summary statistics of the right tree node.
    predict_days : int, optional (default=21)
        The time frame we segment as prediction period

    Returns
    -------
    outliers : a list of (datetime, int) tuple
        A list of outliers, the date and the value for each
    df_pred : pandas DataFrame
        The data set contains actual and predictions for the forecast time frame
    """
    df_pred = forecast[['ds', 'yhat', 'yhat_lower',
                        'yhat_upper']].tail(predict_days)
    df_pred.index = df_pred['ds'].dt.to_pydatetime()
    df_pred.columns = ['ds', 'preds', 'lower_y', 'upper_y']
    df_pred['actual'] = df['y'][today_index: today_index + predict_days].values

    # construct a list of outliers
    outlier_index = list()
    outliers = list()
    for i in range(df_pred.shape[0]):
        actual_value = df_pred['actual'][i]
        if actual_value < df_pred['lower_y'][i] or actual_value > df_pred['upper_y'][i]:
            outlier_index += [i]
            outliers.append((df_pred.index[i], actual_value))
            # optional, print out the evaluation for each outlier
#             print('=====')
#             print('actual value {} fall outside of the prediction interval'.format(actual_value))
#             print('interval: {} to {}'.format(df_pred['lower_y'][i], df_pred['upper_y'][i]))
#             print('Date: {}'.format(str(df_pred.index[i])[:10]))

    return outliers, df_pred


def execute_cross_validation_and_performance_loop(cross_valid_params, metric='mse'):
    """
    Execute Cross Validation and Performance Loop
    
    Args
    ----------
    cross_valid_params: List of dict
      dict value same as cross_validation function argument
      model, horizon, period, initial
    metric: string 
      sort the dataframe in ascending order base on the 
      performance metric of your choice either mse, rmse, mae or mape
    
    Returns
    -------
    A pd.DataFrame with cross_validation result. One row
    per different configuration sorted ascending base on
    the metric inputed by the user.
    """

    assert metric in ['mse', 'rmse', 'mae', 'mape'], \
        'metric must be either mse, rmse, mae or mape'

    df_ps = pd.DataFrame()

    for cross_valid_param in cross_valid_params:
        df_cv = cross_validation(**cross_valid_param)
        df_p = performance_metrics(df_cv, rolling_window=1)
        df_p['initial'] = cross_valid_param['initial']
        df_p['period'] = cross_valid_param['period']
        df_ps = df_ps.append(df_p)

    df_ps = df_ps[['initial', 'horizon', 'period',
                   'mse', 'rmse', 'mae', 'mape', 'coverage']]
    # return df_ps.sort_values(metric)
    print('new')
    df_ps = df_ps.sort_values(metric)
    return df_ps


def prophet(df_dev,
            device,
            parameter='co2',
            begin='2019-03-26',
            end='2019-04-03',
            sampling_period_min=5,
            graph=1,
            predict_day=1,
            interval_width=0.6,
            changepoint_prior_scale=0.01,
            daily_fo=12):
    """
    Combination of df_generator, model, prophet_fit, get_outliers and prophet_plot for randomsearch 
    
    Args:
        - df_dev: Pandas dataframe. Full device-specific dataframe containing:
            o device                                object
            o tenant                                object
            o ds             datetime64[ns, Europe/Zurich]
            o light                                float64
            o temperature                          float64
            o humidity                             float64
            o co2                                  float64
        - device: String. Name of the device written as 'device' + 2-digit number
        - parameter: String. among 'light', 'temperature', 'humidity', 'co2'.
          co2 might be the more "human-activity" related
        - begin: String. Day of the beginning of the new dataframe.
        - end: String. Day of the end of the new dataframe.
        - sampling_period_min: Float. Number of minutes used for the downsampling.
        - graph: Set to None to show the graph and a value if you don't want
          to show the graph.
        - predict_day=1. Float. Number of days predicted. 1 by default.
        - interval_width=0.6. Float. Anomaly threshold. Increase tolerance.
        - changepoint_prior_scale=0.01. Float.  # Adjusting trend flexibility. low = toward
          overfit. Does not make sense to have more than 0.1. Just increasing tolerance*
        - daily_fo=12. Float. Fourier order for the daily seasonality.

    Returns:
        df_p: Pandas dataframe. Performance metric from prophet. Table containing the horizon
          (see Prophet documentation), the mse, rmse, mae, mape, mdape and coverage
        df_pred: Pandas dataframe. Generated by the function get_outliers. The data set contains
        actual and predictions for the forecast time frame.

    return df_p, the performance_metrics of Prophet
    """
    # Convert the sampling period (in min) into string and float for to feed Prophet
    sampling_period_st = str(sampling_period_min) + 'T'
    sampling_period_num = sampling_period_min/60

    # Generate the dataframe analysis
    df, predict_n, today_index, lookback_n = df_generator(
        df_dev,
        device,
        parameter,
        begin,
        end,
        sampling_period_st,
        sampling_period_num,
        graph=graph,
        predict_day=1)

    # config the model
    model = Prophet(interval_width=interval_width,  # anomaly threshold,
                    yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=changepoint_prior_scale)  # Adjusting trend flexibility. should be <0.1 low --> toward overfit
    model.add_seasonality(name='daily', period=1,
                          fourier_order=daily_fo)  # prior scale
#     model.add_seasonality(name='weekly', period=7, fourier_order=15)

    # Fit the model, flag outliers, and visualize
    assert today_index > lookback_n, 'Not enough data for prediction (lookback_n<today_index)'
    plt.close()
    fig, forecast, model = prophet_fit(df, model, today_index, sampling_period_st,
                                       sampling_period_num, lookback_days=lookback_n, predict_days=predict_n)
    outliers, df_pred = get_outliers(
        df, forecast, today_index, predict_days=predict_n)
    prophet_plot(df, fig, today_index,
                 predict_days=predict_n, outliers=outliers)

    # If you want to save the file
    folder = '/Users/path_to_your_folder/figures/'
    name = re.sub("[']", '', str(device)) + '_Prediction_' + \
        str(parameter) + '_' + df.iloc[today_index, 0].strftime('%Y-%m-%d')
    filename = folder + name + '.png'
    plt.savefig(filename, bbox_inches="tight")
    plt.show

    # Cross validation
    df_cv = cross_validation(model, initial='5 days',
                             period='0.5 days', horizon='1 days')
    df_p = performance_metrics(df_cv)
    return df_p, df_pred


def GridSearch_Prophet(prophet_grid, metric='mape'):
    """
    GridSearch tool to determine the optimal parameters for prophet
    
    Args:
        - prophet_grid: List of parameters. Enter it as list(ParameterGrid(prophet_grid)
        - metric: String. Not used yet. May be used to change the metric used to sort
          the tested models.
        
    Return:
        - mape_table: Pandas dataframe. Show the tested parameters and median of Mean  
        Absolute Percentage Error calculated over 1 day.
    """

    # mape_table summarizes the mean of mape according to tested parameters
    mape_table = pd.DataFrame.from_dict(prophet_grid)
    mape_table = mape_table[['device',
                             'parameter',
                             'begin',
                             'end',
                             'sampling_period_min',
                             'interval_width',
                             'daily_fo',
                             'changepoint_prior_scale']]

    mape_table['mape_median'] = np.nan

    # Loop Prophet over the prophet_grid
    a = 0
    for prophet_instance in prophet_grid:
        print('\nprophet_instance nb ' + str(a))
        df_p, df_pred = prophet(**prophet_instance)

        # store the median of mape for 1 day in the table
        mape_table.iloc[a, 8] = df_p.mape.median()
        a += 1

    return mape_table
