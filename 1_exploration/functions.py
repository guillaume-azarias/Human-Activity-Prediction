"""
This script centralized the functions for the data exploration.

The functions in this script are:
* load_ds: Load a device-specific dataframe located in the folder ../data/interim.
If it does not exist, it will start load_raw_data that will create all the
device-specific dataframes.
* load_raw_data(): Load the raw data located in ../data/raw and generate all the
device specific dataframes in the folder ./data/interim
* df_dev_formater: Load a csv file containing the data from a single device
  and return a pandas dataframe where the date and time column is formatted
  in datetime for Switzerland.
* df_generator: show a dataframe. Complete function used for the prophet script
* csv_and_plot_saving: Generates dataframes and plots for specific devices,
  parameters, days and time.
* loop_graph: This function is generating dataframes per device, from day and
start_time for a specific duration.
* plot_scatter_flex: This function is generating a png plot from df_small, for a
specific parameter col and save it name.
* day_night_csv: This function generates the df_night and df_day dataframe for all
parameters from the time they go to bed to the time they wake up.
* df_dev_generator:  This function is to save device-specific dataframe for processing
  in the notebook Prophet_Prediction.ipynb.
"""

from datetime import datetime, timedelta
from pytz import timezone
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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


def load_raw_data():
    '''
    This is a script to generate device-specific dataframes from the csv files we have received.
    
    This script is started if a device-specific dataframe is not found.
    
    This script expect to find the following files in the folder 
        * Raw_data_part1.csv
        * Raw_data_part2.csv
        * Raw_data_part3.csv
        * Raw_data_part4.csv
        * Raw_data_part5.csv
    
    It could be optimized by:
        - increasing the flexibility by finding the name of the csv files
        - Using a loop with tqdm and append newly loaded csv to df_raw (might be computationaly more expensive).
    '''
    print('Loading the raw data...\nThis may take a while.')
    try:
        # Concatenate the csv files
        start_time = time.time()

        print('Status [     ]')
        part1 = pd.read_csv('../data/raw/Copy of '
                            'Raw_data_part1.csv',
                            delimiter=',')
        print('       [#    ]')
        part2 = pd.read_csv('../data/raw/Copy of '
                            'Raw_data_part2.csv',
                            delimiter=',')
        print('       [##   ]')
        part3 = pd.read_csv('../data/raw/Copy of '
                            'Raw_data_part3.csv',
                            delimiter=',')
        print('       [###  ]')
        part4 = pd.read_csv('../data/raw/Copy of '
                            'Raw_data_part4.csv',
                            delimiter=',')
        print('       [#### ]')
        part5 = pd.read_csv('../data/raw/Copy of '
                            'Raw_data_part5.csv',
                            delimiter=',')
        print('       [#####]')
        end_time = time.time()

        print('Data loaded in ' + str(int(end_time - start_time)) +
              " seconds.\nConcatenating the files.")

        start_time_conc = time.time()
        df_raw = pd.concat(
            [part1, part2, part3, part4, part5], ignore_index=True)
        end_time = time.time()
        print('Concatenation done in ' +
              str(int(end_time - start_time_conc)) + " seconds.")

        # Make a list of the devices
        print('\nMake a list of the devices')
        device_list = df_raw['device'].unique()
        device_list.sort()

        # Generate the device-specific dataframes
        print('\nGenerate the device-specific dataframes')
        df_dev_generator(df_raw, device_list)
        print('All device-specific dataframes successfully saved as csv.')
        print('You may restart your kernel and load a device-specific dataframe to save memory.')

    except:
        print('Warning: Raw data not found.')
        print('This script expects to find the following data in ../data/raw')
        print('* Raw_data_part1.csv')
        print('* Raw_data_part2.csv')
        print('* Raw_data_part3.csv')
        print('* Raw_data_part4.csv')
        print('* Raw_data_part5.csv')


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
        lookback_n = int(today_index*0.99) # 1 week 336

    Note: Real values of predict_n, today_index and lookback_n depend on
          sampling_period_st and sampling_period_num. Wrong indications of
          sampling_period_st or sampling_period_num can lead to wrong predictions.

    TODO: Check if any existing function converts sampling_period_st into
          sampling_period_st and vice-versa. Use a Regex-based function could take
          care of it and avoid miscalculations.
    """

    # Check parameters:
    # print('df_dev: ' + str(df_dev.head(2)))
    # print('device: ' + str(device))
    # print('parameter: ' + str(parameter))
    # print('begin: ' + str(begin))
    # print('end: ' + str(end))
    # print('sampling_period_st: ' + str(sampling_period_st))
    # print('sampling_period_num: ' + str(sampling_period_num))
    # print('graph: ' + str(graph))
    # print('predict_day: ' + str(predict_day))

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
        fig_df, ax = plt.subplots(figsize=(7, 3))
        # Data with original frequency
        df_original.y.plot(label="Original", color='gray', linewidth=1)
        df.y.plot(label="Resampled data", color='black', marker='o',
                  linestyle='dashed', linewidth=0.5, markersize=2)

        myFmt = DateFormatter("%d/%m %H:%M")
        ax.xaxis.set_major_formatter(myFmt)
        plt.xlabel('Time', fontsize=8)
        plt.ylabel(parameter, fontsize=8)
        plt.title(name, fontsize=14)
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


def csv_and_plot_saving(df_full, device_list, col, days_instances, hour):
    """
    Generates dataframes and plots for specific devices, parameters, days and time
    
    Args:
        - df_full: pandas dataframe. dataframe containing the data of the device, col, days, hour of interest
        - device_list: list of devices indicated as string.
        - parameters: list of parameters indicated as string
        - days_instances: list of days instance indicated as string
        - hour: String. Hour formatted as 'HH:MM'
        
    Return:
        Save a csv file containing the data for all the parameters, for the specific devices, days and hours
        Save plots of the data specific parameters, devices, days and hours
    """
    start_time = time.time()
    
    for device in device_list:
        print(device)
        df_dev = df.loc[df['device'] == device]
        for days in days_instances:
            begin_str = begin + ' ' + starting_time
            begin_dt = datetime.strptime(begin_str, '%Y-%m-%d %H:%M')

            # Apply the Swiss time zone.
            # http://pytz.sourceforge.net/#localized-times-and-date-arithmetic
            swiss = timezone('Europe/Zurich')
            # swiss.zone
            begin_dt = swiss.localize(begin_dt)
            end_dt = begin_dt + timedelta(days=1)

            # Filter according to begin and end
            df_small = df_full[(df_full['ds'] >= begin_dt) & (df_full['ds'] <= end_dt)]

            if df_small.shape[0]>0:
                # Save the df_small
                folder = '/Users/path_to_your_folder/data/'
                file_name = folder + str(device) + '_' + str(days) + '_h' + str(hour) + '.csv'
                df_small.to_csv(file_name)

                # Plot parameter-specific data
                for col in parameters:
                    plot_scatter_flex(df_small, device, col, days, hour)

    end_time = time.time()
    print('Looping completed')
    print(end_time - start_time)


def loop_graph(df_full, device_list, col, days_instances, start_time, duration):
    """
    This function is generating a dataframe per device, from day and start_time for a
    specific duration.
    You set the device and parameters, day and hour when it starts and how long 
    is the df.
    The df will collect the data:
        - starting from the day (str) and hour (hh:mm) indicated
        - finishing after the indicated duration (int, hours)
        
    Example: if days = '2019-05-12', start_time = 18:00 and duration = 11, the df
    will collect the data:
        - starting from the 2019-05-12 at 18:00
        - finishing on the 2019-05-13 at 5:00 (to check)
        
    Parameters:
        - df_full: pandas dataframe. dataframe containing the data of the device, col, days, hour of interest
        - device: List of devices indicated as string. Each instance will be 
        looped within plot_scatter_dh_flex(df_small, device, col, days, hour)
        - col: List of parameters indicated as string. Each instance will be 
        looped within plot_scatter_dh_flex(df_small, device, col, days, hour)
        - days: List of days formatted as yyyy-mm-dd indicated as string. Each 
        instance will be looped within plot_scatter_flex(df_small, device, 
        col, days, hour)
        - start_time: String of hh:00. Time of the begin of the df.
        - duration: Integer. Duration in hours of the df.
        
    Return:
        Save a csv file containing the data for all the parameters, for the specific devices, days and hours
        Save plots of the data specific parameters, devices, days and hours
    """
    chrono_start = time.time()

    for device in device_list:
        # df_dev is the device-specific dataframe
        print(device)
        df_dev = df_full.loc[df_full['device'] == device]

        # df_small is the device-specific, time-specific dataframe
        for days in days_instances:
            hour_dt = datetime.strptime(start_time, '%H:%M').time()
            print(days)
            date_dt = datetime.strptime(days, '%Y-%m-%d')
            date_d1 = datetime.combine(date_dt, hour_dt)
            date_d2 = date_d1 + timedelta(hours=duration)

            df_small = df_dev[(df_dev['ts_date'] >= date_d1)
                              & (df_dev['ts_date'] <= date_d2)]

            # If there was data acquired at this time, Save the df_small as csv and run plot_scatter_flex
            if df_small.shape[0] > 0:
                folder = '/Users/path_to_your_folder/data/'
                hour_formatted = re.sub(':', 'h', str(start_hour))
                name = str(device) + '_' + str(days) + '_from_' + \
                    str(hour_formatted) + '_dur' + str(duration) + 'h'
                file_name = folder + name + '.csv'
                df_small.to_csv(file_name)

                for col in parameters:
                    plot_scatter_flex(df_small, name, col)

    chrono_end = time.time()
    print('Looping completed in ' +
          str(int(chrono_end - chrono_start)) + ' seconds.')


def plot_scatter_flex(df_small, name, col):
    """
    This function is generating a png plot from df_small, for a specific parameter col and
    save it name.
    
    Parameters (from loop_graph_per_day_hour_flex):
        - df_small: dataframe. df per device, from day and start_time for a specific duration.
        - name: String. indicates the device, day, starting time and duration.
        - col: String. Specific parameter of the graph.
    """
    name = str(col) + '_' + name
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter('ts_date', col, data=df_small, s=2, color='black')
#     ax.plot(df_small.index.time,df_small[col]); # for scatter uncomment prev line
    last = df_small.shape[0] - 1
    ax.set_xlim(left=df_small.iloc[0, 2],
                right=df_small.iloc[last, 2])  # 2 for ts_date
    ax.tick_params(axis='x', labelrotation=45)

    myFmt = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(myFmt)

    ax.set_title(name)
    ax.set_ylabel(col)
    # Note: if the graph is empty but df_small is normal, verify ax.set_xlim is associated with ts_date

    folder = '/Users/path_to_your_folder/figures/'
    filename = folder + name + '.png'
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def day_night_csv(df_dev, device, day_night_df):
    """
    This function generates the df_night and df_day dataframe for all parameters
    from the time they go to bed to the time they wake up.
    This function was for exploration but not further use in the analysis.
    
    Parameters:
        - device: String indicating the device.
        - day_night_df: Pandas dataframe containing the columns
          day_gotobed(%Y-%m-%d), time_gotobed(%H:%M),
          day_wakeup(%Y-%m-%d), time_wakeup(%H:%M)
          
    Return:
        - save one csv for of the data of night
        - save one csv for of the data of day
    """
    chrono_start = time.time()

    # df_night and df_day are device-specific, time-specific dataframe
    for row in range(day_night_df.shape[0]):
        # Load the Swiss time zone.
        swiss = timezone('Europe/Zurich')

        # Go to bed
        date_dt = datetime.strptime(day_night_df.iloc[row, 0], '%Y-%m-%d')
        hour_dt = datetime.strptime(day_night_df.iloc[row, 1], '%H:%M').time()
        date_d1 = datetime.combine(date_dt, hour_dt)
        date_d1 = swiss.localize(date_d1)

        # Wake up
        date_dt = datetime.strptime(day_night_df.iloc[row, 2], '%Y-%m-%d')
        hour_dt = datetime.strptime(day_night_df.iloc[row, 3], '%H:%M').time()
        date_d2 = datetime.combine(date_dt, hour_dt)
        date_d2 = swiss.localize(date_d2)

        # Go to bed the next day
        if row < day_night_df.shape[0]-1:
            date_dt = datetime.strptime(
                day_night_df.iloc[row+1, 0], '%Y-%m-%d')
            hour_dt = datetime.strptime(
                day_night_df.iloc[row+1, 1], '%H:%M').time()
            date_d3 = datetime.combine(date_dt, hour_dt)
            date_d3 = swiss.localize(date_d3)

        # Extract the day and night df
        df_night = df_dev[(df_dev['ds'] >= date_d1) &
                          (df_dev['ds'] <= date_d2)]

        if row < day_night_df.shape[0]-1:
            df_day = df_dev[(df_dev['ds'] >= date_d2) &
                            (df_dev['ds'] <= date_d3)]

        # Save the df
        folder = '/Users/path_to_your_folder/data/interim/'

        if df_night.shape[0] > 0:
            hour_formatted = re.sub(':', 'h', str(day_night_df.iloc[row, 1]))
            name = str(device) + '_Night_' + \
                str(day_night_df.iloc[row, 0]) + '_from_' + hour_formatted
            file_name = folder + name + '.csv'
            df_night.to_csv(file_name)

        if df_day.shape[0] > 0:
            hour_formatted = re.sub(':', 'h', str(day_night_df.iloc[row, 3]))
            name = str(device) + '_Day_' + \
                str(day_night_df.iloc[row, 2]) + '_from_' + hour_formatted
            file_name = folder + name + '.csv'
            df_day.to_csv(file_name)

    chrono_end = time.time()
    print('csv generation completed in ' +
          str(int(chrono_end - chrono_start)) + ' seconds.')


def df_dev_generator(df_raw, device_list):
    """
    This function is save device-specific dataframe for processing in the notebook Prophet_Prediction.ipynb.
    You set the device list. There is no further filtering.
        
    Parameters:
        - device_list: List of devices indicated as string. Each instance will be 
        looped within plot_scatter_dh_flex(df_small, device, col, days, hour)
    
    Does not return any variable
    """
    chrono_start = time.time()

    for device in device_list:
        # df_dev is the device-specific dataframe
        print(device)
        start_time = time.time()
        df_dev = df_raw.loc[df_raw['device'] == device]

        if df_dev.shape[0] > 0:
            # df_dev = df_dev.drop(['Unnamed: 0'], axis=1)

            folder = '/Users/path_to_your_folder/data/interim'
            name = str(device) + '_alldays'
            file_name = folder + name + '.csv'
            df_dev.to_csv(file_name)
            end_time = time.time()
            print('Processed and saved in ' +
                  str(int(end_time - start_time)) + 'sec.')

    chrono_end = time.time()
    print('Looping completed in ' +
          str(int(chrono_end - chrono_start)) + ' seconds.')
