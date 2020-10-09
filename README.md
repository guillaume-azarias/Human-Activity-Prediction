# Identification of patterns correlating with person's activity

*Capstone project realized during the Data Science bootcamp at the Propulsion Academy (02/2020)*

Collaboration with Christophe Bousquet and Juhyun Schoebi.


## Background:
Human activity patterns can be derived from different sensors like CO2, temperature, noise, etc. If irregularities are detected, an app processing the ambient environmental data can actively notifies caregivers.

## Project Goals : Find patterns in the sensor signals that correlate with a person's activity.

## Objectives :

This project focuses on analysing data derived from sensors deployed in households.

The overall goal is to find patterns in the sensors' signals that correlate with a person's activity.

## Milestones:
**Milestones 1**: Normalize the data, identify patterns, detect certain activities.

**Milestones 2**: Real-time activity reporting every 15 - 30 minutes. Be able to detect an activity (and maybe the type of activity) and send a notification.

## Data:
- device                                object
- tenant                                object
- light                                float64
- temperature                          float64
- humidity                             float64
- co2                                  float64

Light, temperature, humidity and CO2 were time series of ambient value measured every 20sec.

### Approach
**Skills**: Time-series analysis of data. Clustering - pattern analysis

**Output**: Analysis report of activity. Dashboard for a patient-specific overview

**Value Proposition**: Tells the caregiver/family if a resident is behaving normally or exhibit an abnormal activity. Detects persons in need of help

**Integration**: Dashboard accessible through a web interface

**Customers**: Caregivers, family member, person in charge to provide medical help

### Code structure

**`1_exploration`**: first handling of the data. Plotting time-series to find the relevant duration and amplitude of patterns. Data interpretation on a few examples.

**`2_forecasting`**: Anomaly detection using time-series forecasting using Prophet from facebook. Possibility to run a single instance, or let the script determine the best parameters for prediction: it will try several numbers of days for the training (this was the most critical parameter to my experience), sort the results according to the mean average percentage error and save the best Prophet result. 

**`2_forecasting_GridSearch`**: This script aims to check if the script is relevant for production. A loop to run the script on a random sample of days (eg 10 days per device, user-defined), to be able to check if the script works with all kind of results.

**`3_dashboard`**:  Web interface dedicated to the caregivers to inform and alert on the patient activities.
