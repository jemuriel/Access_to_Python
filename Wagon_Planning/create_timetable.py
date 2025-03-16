from Wagon_Planning import New_Time_Table_Run
from Wagon_Planning.New_Time_Table_Run import NewTimeTable
import os

os.chdir(r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning\DataFiles\Wagon_Balancing"
         r"\Model_Files\MBM_Corridor")

timetable_file = "6_train_services.csv"
wagon_plan_file = "7.1_wagon_plan_2025.csv"
wagon_mapping = "8_wagon_mapping.csv"
inventory_output = "inventory_levels.csv"
adjustments_output = "inventory_adjustments.csv"
wagon_output = '9_wagon_config_transformed_MBM.csv'


def create_timetable() -> NewTimeTable:
    train_timetable = NewTimeTable(30, timetable_file, wagon_plan_file, wagon_mapping, inventory_output,
                                   adjustments_output, wagon_output, 1)
    return train_timetable

# train_timetable = create_timetable()
# # train_timetable.
# print()