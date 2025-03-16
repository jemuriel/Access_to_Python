import os
import time

import pandas as pd

from Wagon_Planning.Time_Table_Run import TimeTable

# data_file = "C:\\Users\\61432\\OneDrive - Pacific National\\Tactical_Train_Planning\\DataFiles\\OutboundFile_No8.csv"
# data_file = "C:\\Users\\61432\\OneDrive - Pacific National\\Tactical_Train_Planning\\DataFiles\\outboundFile_No8.csv"
# forecast_file = "C:\\Users\\61432\\OneDrive - Pacific National\\Tactical_Train_Planning\\DataFiles\\PN_Forecast.csv"
# output_forecast = "C:\\Users\\61432\\OneDrive - Pacific National\\Tactical_Train_Planning\\DataFiles\\forecasted_flows.csv"
# box_file = "C:\\Users\\61432\\OneDrive - Pacific National\\Tactical_Train_Planning\\DataFiles\\box_types.csv"

# Process the files and calculate the weights probabilistically
# print('Starting Probabilistic Model')
# new_model = Probabilistic_Model(data_file, forecast_file, box_file, output_forecast)
# new_model.run_complete_model()
# new_model.run_process_data()

# -----------------------------------------------------------------------------------------------------------------
# WAGON BALANCING MODEL
# -----------------------------------------------------------------------------------------------------------------
inventory_output = "C:\\Users\\61432\\OneDrive - Pacific National\\Tactical_Train_Planning\\DataFiles\\Wagon_Balancing\\inventory_levels.csv"
adjustments_output = "C:\\Users\\61432\\OneDrive - Pacific National\\Tactical_Train_Planning\\DataFiles\\Wagon_Balancing\\inventory_adjustments.csv"

timetable_file = "C:\\Users\\61432\\OneDrive - Pacific National\\Tactical_Train_Planning\\DataFiles\\Wagon_Balancing\\train_services.csv"
wagon_plan_file = "C:\\Users\\61432\\OneDrive - Pacific National\\Tactical_Train_Planning\\DataFiles\\Wagon_Balancing\\wagon_plan.csv"

access_db = r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning\Sample Train Plan.accdb"

wagon_balance = TimeTable(30, access_db, inventory_output, adjustments_output,1)
# wagon_balance.run_one_timetable()

# wagon_balance = TimeTable(30, access_db,  inventory_output, adjustments_output, 1)
# wagon_balance.run_one_timetable()


# Get the current working directory
# current_directory = os.getcwd()
# print("Current Working Directory:", current_directory)