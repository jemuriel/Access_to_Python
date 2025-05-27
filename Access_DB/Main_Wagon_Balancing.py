import os

import pandas as pd

from Forecast_Disag.Improved_Prob_Model import Probabilistic_Model
from Wagon_Planning.New_Time_Table_Run import NewTimeTable
from Wagon_Planning.Wagon_Assigner import WagonAssigner

os.chdir(r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning\DataFiles\Wagon_Balancing"
         r"\Model_Files\MBM_corridor")
# -----------------------------------------------------------------------------------------------------------------
# FORECAST FILES
# -----------------------------------------------------------------------------------------------------------------
data_file = pd.read_csv(r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning\DataFiles\outboundFile_No8.csv")
# forecast_file = pd.read_csv('../csv_files/4_PN_Forecast.csv')
forecast_file = pd.read_csv(r"C:\Users\61432\Downloads\forecast_versions_rows.csv")

output_forecast = (r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning\DataFiles\Wagon_Balancing"
                   r"\Model_Files\MBM_Corridor\5_disag_forecasted_flows_MBM_new.csv")
box_file = pd.read_csv(r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning\DataFiles\box_types.csv")

# Process the files and calculate the weights probabilistically
print('Starting Probabilistic Model')
# new_model = Probabilistic_Model(data_file, forecast_file, box_file, output_forecast)
# disag_forecast = new_model.run_complete_model()

# Read from csv temporarily
# disag_forecast = pd.read_csv("5_disag_forecasted_flows_MBM.csv")

# -----------------------------------------------------------------------------------------------------------------
# WAGON BALANCING FILES
# -----------------------------------------------------------------------------------------------------------------
inventory_output = (r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning\DataFiles\Wagon_Balancing"
                    r"\Model_Files\wagon_balancing\inventory_levels.csv")
adjustments_output = (r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning\DataFiles\Wagon_Balancing"
                      r"\Model_Files\wagon_balancing\inventory_adjustments.csv")
wagon_output = (r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning\DataFiles\Wagon_Balancing"
                r"\Model_Files\wagon_balancing\9_wagon_config_transformed_MBM.csv")

timetable_file = "6_train_services.csv"

# ===========================================================================
# wagon_plan_file = "7_wagon_plan.csv"
wagon_plan = pd.read_csv("7.1_wagon_plan_2025.csv")
# ============================================================================

wagon_mapping = "8_wagon_mapping.csv"

train_timetable = NewTimeTable(30, timetable_file, wagon_plan, wagon_mapping)
inventory_levels, inventory_adjustments = train_timetable.balance_inventory()

inventory_levels.to_csv(inventory_output, index=False)

print()

# UI interface ---------------------------------------------


# -----------------------------------------------------------------------------------------------------------------
# EVALUATE EFFICIENCY FOR EVERY TRAIN SERVICE
# -----------------------------------------------------------------------------------------------------------------
train_summary = '10_train_summary_MBM.csv'
unassigned_boxes = '11_unassigned_containers_MBM.csv'
wagon_assignments = '11.1_wagon_assignments_MBM.csv'
WagonAssigner.run_yearly_assignment(disag_forecast, train_timetable, train_summary, wagon_assignments, unassigned_boxes)


# -----------------------------------------------------------------------------------------------------------------
# COMPARE TRAIN LEGS
# -----------------------------------------------------------------------------------------------------------------

def compare_forecast_legs():
    # Safely create 'ORIGIN' and 'DESTINATION' columns
    filtered_forecast = disag_forecast[['TRAIN_NUM', 'OD_PAIR']].drop_duplicates()
    filtered_forecast[['ORIGIN', 'DESTINATION']] = filtered_forecast['OD_PAIR'].str.split('-', expand=True)

    # Group the forecast by 'TRAIN_NUM'
    grouped_forecast = filtered_forecast.groupby('TRAIN_NUM')

    # Collect train services information for easier lookup
    train_services_values = set(train_timetable.train_services_dic.keys())

    # Prepare train data
    train_data = []

    # Loop through each group (unique train number) and rows within each group
    for train_num, train_forecast in grouped_forecast:
        for idx, row in train_forecast.iterrows():
            # Initialize access_leg as None
            access_leg = None
            leg_id = None

            # Check if the train is in the train services dictionary
            if train_num in train_services_values:
                train_legs = train_timetable.train_services_dic.get(train_num, [])
                # Iterate over train legs to find the corresponding access leg
                for the_leg in train_legs:
                    the_leg_details = train_legs.get(the_leg, None)
                    if the_leg_details and (
                            row['ORIGIN'] == the_leg_details.leg_origin_terminal
                            and row['DESTINATION'] == the_leg_details.leg_destination_terminal
                    ):
                        access_leg = row['OD_PAIR']
                        leg_id = the_leg
                        break

            # Append the resulting data
            train_data.append({
                'FORECAST_TRAIN_NUMBER': train_num,
                'FORECAST_LEG_OD': row['OD_PAIR'],
                'ACCESS_DB_TRAIN_NUMBER': train_num if train_num in train_services_values else None,
                'ACCESS_DB_LEG_OD': access_leg,
                'LEG_NUMBER': leg_id
            })

    # Create a DataFrame from the collected train data and write to CSV
    train_pd = pd.DataFrame(train_data)
    train_pd.to_csv('13_Forecast_leg_comparison_MBM.csv', index=False)


def matching_train_services():
    # Safely create 'ORIGIN' and 'DESTINATION' columns
    filtered_forecast = disag_forecast[['TRAIN_NUM', 'OD_PAIR']].drop_duplicates()
    filtered_forecast[['ORIGIN', 'DESTINATION']] = filtered_forecast['OD_PAIR'].str.split('-', expand=True)

    # Collect train services information for comparison
    train_services_dic = train_timetable.train_services_dic

    # Prepare train data
    train_data = []

    # Loop through each entry in the train services dictionary
    for train_num, train_legs in train_services_dic.items():
        for the_leg in train_legs:
            # Obtain leg details from the dictionary
            the_leg_details = train_legs.get(the_leg, None)
            if not the_leg_details:
                continue

            # Filter forecast data to find matching origin and destination
            matching_forecast = filtered_forecast[
                (filtered_forecast['TRAIN_NUM'] == train_num) &
                (filtered_forecast['ORIGIN'] == the_leg_details.leg_origin_terminal) &
                (filtered_forecast['DESTINATION'] == the_leg_details.leg_destination_terminal)
                ]

            # Append matched data to train_data list
            for _, row in matching_forecast.iterrows():
                train_data.append({
                    'ACCESS_DB_TRAIN_NUMBER': train_num,
                    'ACCESS_DB_LEG_OD': row['OD_PAIR'],
                    'FORECAST_TRAIN_NUMBER': row['TRAIN_NUM'],
                    'FORECAST_LEG_OD': row['OD_PAIR']
                })

    # Create a DataFrame from the collected train data and write to CSV
    train_pd = pd.DataFrame(train_data)
    train_pd.to_csv('14_matching_trains_MBM.csv', index=False)


def access_comparisson():
    # Safely create 'ORIGIN' and 'DESTINATION' columns
    filtered_forecast = disag_forecast[['TRAIN_NUM', 'OD_PAIR']].drop_duplicates()
    filtered_forecast[['ORIGIN', 'DESTINATION']] = filtered_forecast['OD_PAIR'].str.split('-', expand=True)

    # Collect train services information for comparison
    train_services_dic = train_timetable.train_services_dic

    # Prepare train data
    train_data = []

    # Loop through each entry in the train services dictionary
    for train_num, train_legs in train_services_dic.items():
        # print(f"Processing Train Number: {train_num}")
        for the_leg in train_legs:
            # Obtain leg details from the dictionary
            the_leg_details = train_legs.get(the_leg, None)
            if not the_leg_details:
                continue

            # print(
            #     f"  Train Leg: {the_leg} (Origin: {the_leg_details.leg_origin_terminal}, Destination: {the_leg_details.leg_destination_terminal})")

            # Filter forecast data to find matching origin and destination
            matching_forecast = filtered_forecast[
                (filtered_forecast['TRAIN_NUM'] == train_num) &
                (filtered_forecast['ORIGIN'] == the_leg_details.leg_origin_terminal) &
                (filtered_forecast['DESTINATION'] == the_leg_details.leg_destination_terminal)
                ]

            # If there are matching rows, append them
            if not matching_forecast.empty:
                for _, row in matching_forecast.iterrows():
                    train_data.append({
                        'ACCESS_DB_TRAIN_NUMBER': train_num,
                        'LEG_NUMBER': the_leg,
                        'ACCESS_DB_LEG_OD': row['OD_PAIR'],
                        'FORECAST_TRAIN_NUMBER': row['TRAIN_NUM'],
                        'FORECAST_LEG_OD': row['OD_PAIR']
                    })
            else:
                # If there are no matches, append the train and leg information with None for forecast-related fields
                train_data.append({
                    'ACCESS_DB_TRAIN_NUMBER': train_num,
                    'ACCESS_DB_LEG_OD': f"{the_leg_details.leg_origin_terminal}-{the_leg_details.leg_destination_terminal}",
                    'LEG_NUMBER': the_leg,
                    'FORECAST_TRAIN_NUMBER': None,
                    'FORECAST_LEG_OD': None,
                })

    # Create a DataFrame from the collected train data and write to CSV
    train_pd = pd.DataFrame(train_data)
    train_pd.to_csv('12_Access_comparison_MBM.csv', index=False)


compare_forecast_legs()
matching_train_services()
access_comparisson()
