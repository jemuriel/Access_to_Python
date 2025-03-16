from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Wagon_Planning.Util_Classes import Terminal
# from Wagon_Planning.Train_Leg import Train_Leg
from Access_DB.ConnectDB import ReadAccessDB


class TimeTable:
    def __init__(self, time_buckets, access_db, inventory_output, adjustments_output, num_weeks):
        self.num_weeks = num_weeks
        self.inventory_output = inventory_output
        self.adjustments_output = adjustments_output
        self.access_db = access_db
        self.time_buckets = time_buckets

        # self.wagon_pool = {}  # {Wagon_Class:Wagon}

        self.time_range = range(0, self.num_weeks * 7 * 24 * 60, self.time_buckets)  # 10-minute intervals for one week

        db_reader = ReadAccessDB(self.access_db)
        self.train_services_df = db_reader.read_table('tblTrainTimesNewSingleWeek')
        self.train_services_dic = {}
        self.wagon_plan = db_reader.read_table('qryWagon_Plan')

        self.initialise_train_services()

        self.wagon_types = set(self.wagon_plan['WAGON_CLASS'])

        self.terminals = self.create_terminals()

        self.timeline_inventory = self.create_inventory_structure()
        self.adjustment_recorder = self.create_inventory_structure()

    def initialise_train_services(self):
        """Initialize the train services and configuration based on the type."""

        # Sort each train service by legs
        def sort_legs(df_group):
            """Sort legs in order to find the origin and destination of the service."""

            # Drop rows with missing or identical 'Departing Location' and 'Arriving Location' values
            df_group = df_group.dropna(
                subset=['Departing Location', 'Arriving Location', 'Departure Day', 'Arrival Day'])
            df_group = df_group[df_group['Departing Location'] != df_group['Arriving Location']].copy()

            # Return immediately if there's only one leg
            if len(df_group) == 1:
                df_group['Leg_Number'] = 1
                return df_group.reset_index(drop=True)

            # Identify origin (a 'Departing Location' that is not an 'Arriving Location')
            all_origins = df_group['Departing Location'].tolist()
            all_destinations = df_group['Arriving Location'].tolist()
            origin = next((loc for loc in all_origins if loc not in all_destinations), None)

            if origin is None:
                # If no unique origin is found, return as-is with leg numbers reset
                df_group['Leg_Number'] = range(1, len(df_group) + 1)
                return df_group.reset_index(drop=True)

            # Initialize ordered rows and set the starting origin
            ordered_rows = []
            next_origin = origin
            leg_number = 1
            processed_indices = set()

            # Sort the legs by chaining 'Departing Location' to 'Arriving Location'
            for _ in range(len(df_group)):
                # Find rows with the current origin as 'Departing Location'
                row = df_group[df_group['Departing Location'] == next_origin]

                if row.empty:
                    # Add remaining unprocessed rows with incremental leg numbers
                    remaining_rows = df_group[~df_group.index.isin(processed_indices)]
                    remaining_rows = remaining_rows.assign(
                        Leg_Number=range(leg_number, leg_number + len(remaining_rows)))
                    ordered_rows.extend(remaining_rows.to_dict(orient='records'))
                    break

                # Process each matching row
                row = row.assign(Leg_Number=leg_number)
                for _, row_data in row.iterrows():
                    ordered_rows.append(row_data.to_dict())
                    processed_indices.add(row_data.name)
                    next_origin = row_data['Arriving Location']

                leg_number += 1

            # Convert ordered rows to DataFrame
            ordered_df = pd.DataFrame(ordered_rows).reset_index(drop=True)
            return ordered_df

        self.train_services_df['Primary Train'] = self.train_services_df['Primary Train'].str.upper()
        # sorted_df = self.train_services_df.groupby('Primary Train').apply(sort_legs).reset_index(drop=True)

        sorted_df = pd.DataFrame()
        for train, group in self.train_services_df.groupby('Primary Train'):
            sorted_df = pd.concat([sorted_df, sort_legs(group)])

        # File with the missing services that were not sorted - JUST USE AS A CHECK-UP
        missing_rows = self.train_services_df[
            ~self.train_services_df.isin(sorted_df.to_dict(orient='list')).all(axis=1)]
        # missing_rows = pd.concat([self.train_services_df, sorted_df]).drop_duplicates(keep=False)
        # missing_rows.to_csv(self.rows_not_sorted, index=False)

        # Exclude trains in the trains_no_more file
        # sorted_df = sorted_df[~sorted_df['Primary Train'].isin(self.trains_no_more)]

        self.train_services_df = sorted_df
        self.train_services_df = self.train_services_df.drop('Cut Off Time', axis=1)
        self.train_services_df.columns = ['TRAIN_NAME', 'TRAIN', 'ORIGIN', 'DEP_TIME', 'DEP_DAY', 'ARR_TIME', 'ARR_DAY',
                                          'DESTINATION', 'LEG_NUM']
        self.train_services_df['DEP_TIME'] = pd.to_datetime(self.train_services_df['DEP_TIME'], dayfirst=True,
                                                            errors='coerce')
        self.train_services_df['ARR_TIME'] = pd.to_datetime(self.train_services_df['ARR_TIME'], dayfirst=True,
                                                            errors='coerce')
        self.train_services_df['DEP_TIME'] = self.train_services_df['DEP_TIME'].dt.time
        self.train_services_df['ARR_TIME'] = self.train_services_df['ARR_TIME'].dt.time

        self.train_services_df['ORIGIN'] = self.train_services_df['ORIGIN'].str.upper()
        self.train_services_df['DESTINATION'] = self.train_services_df['DESTINATION'].str.upper()

        long_format_df = self.change_to_long_format(self.train_services_df)

        long_format_df.to_csv(
            "C:\\Users\\61432\\OneDrive - Pacific National\\Tactical_Train_Planning\\DataFiles\\Wagon_Balancing\\filtered_long_train_services.csv",
            index=False)

        self.train_services_dic = self.create_train_services()
        self.create_current_config()

    def change_to_long_format(self, df):
        # Weekday mapping
        weekday_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }

        # Base date (first Sunday is 5 Jan 2025)
        base_date = datetime(2025, 1, 5)

        # Calculate full datetime values
        departure_datetimes = []
        arrival_datetimes = []

        for idx, row in df.iterrows():
            dep_day_offset = (weekday_map[row['DEP_DAY']] - base_date.weekday()) % 7
            arr_day_offset = (weekday_map[row['ARR_DAY']] - base_date.weekday()) % 7

            dep_date = base_date + timedelta(days=dep_day_offset)
            arr_date = base_date + timedelta(days=arr_day_offset)

            dep_time_obj = datetime.strptime(str(row['DEP_TIME']), '%H:%M:%S').time()
            arr_time_obj = datetime.strptime(str(row['ARR_TIME']), '%H:%M:%S').time()

            dep_datetime = datetime.combine(dep_date, dep_time_obj)
            arr_datetime = datetime.combine(arr_date, arr_time_obj)

            if arr_datetime < dep_datetime:
                arr_datetime += timedelta(days=1)

            departure_datetimes.append(dep_datetime)
            arrival_datetimes.append(arr_datetime)

        df['DEP_DATETIME'] = departure_datetimes
        df['ARR_DATETIME'] = arrival_datetimes

        # Prepare output table
        output_df = df[['TRAIN_NAME', 'TRAIN', 'ORIGIN', 'LEG_NUM']].copy()
        output_df['DEP_TIME'] = df['DEP_DATETIME']
        output_df['ARR_TIME'] = np.nan  # initialize

        # Apply logic: first leg ARR_TIME = null, others = previous ARR_DATETIME
        grouped_df = df.groupby('TRAIN_NAME')

        for train, train_group in grouped_df:
            train_group = train_group.sort_values('LEG_NUM')

            for _, row in train_group.iterrows():
                leg_num = row['LEG_NUM']
                if leg_num == 1:
                    output_df.loc[(output_df['TRAIN_NAME'] == train) & (output_df['LEG_NUM'] == leg_num), 'ARR_TIME'] = pd.NaT
                else:
                    prev_leg = leg_num - 1
                    prev_arrival = train_group.loc[train_group['LEG_NUM'] == prev_leg, 'ARR_DATETIME'].iloc[0]
                    output_df.loc[
                        (output_df['TRAIN_NAME'] == train) & (output_df['LEG_NUM'] == leg_num), 'ARR_TIME'] = prev_arrival

            # Add final row for MFT arrival only
            last_leg = train_group.iloc[-1]
            final_row = {
                'TRAIN_NAME': last_leg['TRAIN_NAME'],
                'TRAIN': '1bm2' + last_leg['DESTINATION'],
                'ORIGIN': last_leg['DESTINATION'],
                'ARR_TIME': last_leg['ARR_DATETIME'],
                'DEP_TIME': np.nan,
                'LEG_NUM': last_leg['LEG_NUM'] + 1
            }
            output_df = pd.concat([output_df, pd.DataFrame([final_row])], ignore_index=True)

        output_df = output_df.sort_values(['TRAIN_NAME','LEG_NUM']).reset_index(drop=True)

        # Convert before formatting
        output_df['ARR_TIME'] = pd.to_datetime(output_df['ARR_TIME'], errors='coerce')
        output_df['DEP_TIME'] = pd.to_datetime(output_df['DEP_TIME'], errors='coerce')

        # Format as string for display
        output_df['ARR_TIME'] = output_df['ARR_TIME'].apply(lambda x: x.strftime('%d/%m/%Y %H:%M') if pd.notnull(x) else '')
        output_df['DEP_TIME'] = output_df['DEP_TIME'].apply(lambda x: x.strftime('%d/%m/%Y %H:%M') if pd.notnull(x) else x)

        # Final column order
        output_df = output_df[['TRAIN_NAME', 'TRAIN', 'ORIGIN', 'ARR_TIME', 'DEP_TIME', 'LEG_NUM']]

        filtered_df = output_df[output_df['TRAIN_NAME'].str.contains('MB|BM|MP|PM|MS|SM|MA|AM', na=False)]
        filtered_df['CORRIDOR'] = np.where(filtered_df['TRAIN_NAME'].str.contains('MB|BM|MS|SM', na=False), 'East', 'West')

        return filtered_df


    def create_inventory_structure(self):
        """Create an empty inventory structure for terminals and wagon types."""
        return {terminal: {wagon_type: [] for wagon_type in self.wagon_types} for terminal in
                self.terminals}

    def create_train_services(self):
        train_services = {}
        for _, row in self.train_services_df.iterrows():
            train_num = row['TRAIN_NAME']
            leg_num = row['LEG_NUM']
            the_leg = Train_Leg(row['ORIGIN'], row['DESTINATION'], row['DEP_DAY'], row['DEP_TIME'],
                                row['ARR_DAY'],
                                row['ARR_TIME'])

            if train_num not in train_services:
                train_services[train_num] = {}

            train_services[train_num][leg_num] = the_leg

        return train_services

    # Creates a random based on planned train service (static)
    def create_current_config(self):
        # number_configurations = {row['Train_Num'] for _, row in filename.iterrows()}
        self.wagon_plan.columns = ['TRAIN_NAME', 'TRAIN', 'ORIGIN', 'SERVICE', 'WAGON_CLASS', 'NUM_WAGONS', 'DEP_DAY']
        self.wagon_plan = self.wagon_plan[['TRAIN_NAME', 'TRAIN', 'ORIGIN', 'SERVICE', 'WAGON_CLASS', 'NUM_WAGONS']]
        self.wagon_plan['TRAIN_NAME'] = self.wagon_plan['TRAIN_NAME'].str.upper()

        self.wagon_plan['ORIGIN'] = self.wagon_plan['ORIGIN'].str.upper()

        # Exclude trains in the trains_no_more file
        # self.wagon_plan = self.wagon_plan[~self.wagon_plan['TRAIN_NAME'].isin(self.trains_no_more)]

        # Trains in train services that are not in wagon plan ----------------------------------------------------------
        temp_trains = self.train_services_df['TRAIN_NAME'].unique()
        temp_wagon = self.wagon_plan['TRAIN_NAME'].unique()
        missing_trains = np.setdiff1d(temp_trains, temp_wagon)
        missing_trains_df = pd.DataFrame(missing_trains, columns=['TRAIN_NAME'])
        # missing_trains_df.to_csv(self.trains_without_wagons, index=False)`

        # missing_trains = [train for train in temp_trains if train not in temp_wagon]
        print(f'Train services not in wagon plan: {len(missing_trains)}')
        # ---------------------------------------------------------------------------------------------------------------

        # get the leg number from train services
        temp_train_ser = self.train_services_df[['TRAIN_NAME', 'TRAIN', 'ORIGIN', 'LEG_NUM']].drop_duplicates()
        self.wagon_plan = self.wagon_plan.merge(temp_train_ser, on=['TRAIN_NAME', 'TRAIN', 'ORIGIN'], how='left')

        # self.wagon_plan.to_csv("C:\\Users\\61432\\OneDrive - Pacific National\\Tactical_Train_Planning\\DataFiles\\Wagon_Balancing\\wagon_plan.csv",
        #                        index=False)

        grouped_rows = self.wagon_plan.groupby('TRAIN_NAME')

        def update_wagon_number():
            # Iterate over each train service
            for train_number, leg_dictionary in self.train_services_dic.items():
                if train_number in missing_trains:
                    continue

                # Filter rows for the current train number
                relevant_rows = grouped_rows.get_group(train_number)

                # Iterate over each row for the current train number
                for _, row in relevant_rows.iterrows():
                    wagon_class = row['WAGON_CLASS']
                    leg_num = row['LEG_NUM']
                    num_wagons = row['NUM_WAGONS']
                    leg_wagon_configuration = leg_dictionary[leg_num].wagon_configuration

                    # Update the wagon configuration for the specific leg
                    leg_wagon_configuration[wagon_class] = leg_wagon_configuration.get(wagon_class, 0) + num_wagons

        update_wagon_number()

    def create_terminals(self):
        terminals = set(self.train_services_df['ORIGIN']).union(self.train_services_df['DESTINATION'])
        terminals_dict = {terminal_name: Terminal(terminal_name) for terminal_name in terminals}

        # Set the wagon inventory
        my_wagon_inventory = {the_wagon: 0 for the_wagon in self.wagon_types}

        for the_terminal in terminals_dict.values():
            the_terminal.wagon_inventory = my_wagon_inventory

        # [setattr(the_terminal,'wagon_inventory',my_wagon_inventory) for the_terminal in terminals_dict.values()]

        return terminals_dict

    @staticmethod
    def min_to_weeknum(minutes):
        minutes_per_week = 7 * 24 * 60  # 10080 minutes in a week
        week_number = minutes // minutes_per_week
        return week_number

    @staticmethod
    def is_within_time_bucket(time_table, t, week_num, time_buckets):
        # Convert week to minutes and calculate bucket boundaries
        start_time = t - time_buckets
        end_time = t
        min_time_table = time_table + week_num * 7 * 24 * 60
        return start_time < min_time_table <= end_time

    @staticmethod
    def remove_zero_values(the_dictionary):
        keys_to_remove = [
            (terminal, wagon_key)
            for terminal, wagon_dict in the_dictionary.items()
            for wagon_key, record_list in wagon_dict.items()
            if sum(inventory for _, _, inventory in record_list) == 0
        ]
        for terminal, wagon_key in keys_to_remove:
            del the_dictionary[terminal][wagon_key]

    def balance_inventory(self):
        # Initialize wagon inventory for all terminals and wagon types
        wagon_inventory = {terminal: {wagon_type: 0 for wagon_type in self.wagon_types}
                           for terminal in self.terminals}

        def update_inventory(terminal, wagon_config, adjustment):
            """Update the inventory based on wagon configuration and adjustment."""
            if TimeTable.is_within_time_bucket(
                    train_leg.dep_min_num if adjustment < 0 else train_leg.arr_min_num, t,
                    TimeTable.min_to_weeknum(t), self.time_buckets):
                for wagon_type, wagon_number in wagon_config.items():
                    wagon_inventory[terminal][wagon_type] += adjustment * wagon_number

                    self.adjustment_recorder[terminal][wagon_type].append(
                        (TimeTable.min_to_weeknum(t), t, train_number, leg_number,
                         train_leg.leg_origin_terminal, train_leg.leg_destination_terminal,
                         train_leg.dep_min_num, train_leg.arr_min_num,
                         adjustment * wagon_number))
                    # the_train_service.running = True

        for t in self.time_range:
            # for t in tqdm(self.time_range, desc="Running weekly schedule", unit="time"):
            for train_number, the_leg in self.train_services_dic.items():
                for leg_number, train_leg in the_leg.items():
                    # if not the_train_service.running:
                    # Update inventory for departures
                    update_inventory(train_leg.leg_destination_terminal, train_leg.wagon_configuration, 1)

                    # Update inventory for arrivals
                    update_inventory(train_leg.leg_origin_terminal, train_leg.wagon_configuration, -1)

            # Record inventory levels at this time bucket
            for terminal in self.terminals:
                for wagon_type in self.wagon_types:
                    self.timeline_inventory[terminal][wagon_type].append(
                        (TimeTable.min_to_weeknum(t), t, wagon_inventory[terminal][wagon_type]))

        TimeTable.remove_zero_values(self.timeline_inventory)

    # Convert inventory_over_time to a DataFrame for CSV export
    def convert_to_data_frame(self):
        rows = []
        more_rows = []

        for terminal in self.terminals:
            new_wagon_types = set(wagon_type for wagon_type, record_list in self.timeline_inventory[terminal].items())

            for wagon_type in new_wagon_types:
                # Inventory timeline file
                for week, t, inventory in self.timeline_inventory[terminal][wagon_type]:
                    # time_label = f"{t // (24 * 60)}d {(t % (24 * 60)) // 60:02}:{t % 60:02}"
                    day = f"{t // (24 * 60)}"
                    time_label = f"{(t % (24 * 60)) // 60:02}:{t % 60:02}"

                    rows.append({
                        'Week': week + 1,
                        'Day': day,
                        'Time': t,
                        'Day_Time': time_label,
                        'Terminal': terminal,
                        'Wagon Type': wagon_type,
                        'Inventory': inventory
                    })

                # Inventory adjustment file

                for (week, t, train_number, leg_number, origin_leg, destination_leg,
                     dep_min_num, arr_min_num, adjustment) in self.adjustment_recorder[terminal][wagon_type]:
                    # time_label = f"{t // (24 * 60)}d {(t % (24 * 60)) // 60:02}:{t % 60:02}"
                    day = f"{t // (24 * 60)}"
                    time_label = f"{(t % (24 * 60)) // 60:02}:{t % 60:02}"

                    more_rows.append({
                        'Week': week + 1,
                        'Day': day,
                        'Time': t,
                        'Day_Time': time_label,
                        'Terminal': terminal,
                        'Wagon Type': wagon_type,
                        'Train_number': train_number,
                        'Leg_number': leg_number,
                        'Origin_leg': origin_leg,
                        'Destination_leg': destination_leg,
                        'Departure_min': dep_min_num,
                        'Arrival_min': arr_min_num,
                        'Adjustment': adjustment
                    })

        inventory_df = pd.DataFrame(rows)

        adjustment_df = pd.DataFrame(more_rows)

        self.write_or_append(inventory_df, self.inventory_output)
        self.write_or_append(adjustment_df, self.adjustments_output)
        # self.write_to_excel(inventory_df, self.file_output, "inventory_balance")
        # self.write_to_excel(adjustment_df, self.file_output, "inventory_adjustments")

    def plot_results(self, timeline_inventory):
        # Iterate over each terminal to create a separate plot
        for terminal in self.terminals:
            fig, ax = plt.subplots(figsize=(14, 8))

            for wagon_type in self.wagon_types:
                # Extract time and inventory data for the current terminal and wagon type
                total_inventory = [(week, t, inventory) for week, t, inventory in
                                   timeline_inventory[terminal][wagon_type]]

                # Format the time to show the week number and day name with half-hour intervals
                start_date = pd.Timestamp(f'2023-12-31')  # Sunday at 00:00
                # times = [f"Week {week} - {(pd.Timestamp(f'2023-12-31') + pd.to_timedelta(t, 'm')).strftime('%a %H:%M')}"
                # times = [f"Week {week} - {(pd.Timestamp(f'2023-12-31') + pd.to_timedelta(t, 'm')).floor('H').strftime('%a %H:%M')}"
                times = [f"Week {week} - {(start_date + pd.to_timedelta(t, 'm')).floor('4H').strftime('%a %H:%M')}"

                         for week, t, _ in total_inventory]
                inventories = [inventory for _, _, inventory in total_inventory]

                # Plot the inventory over time for this terminal and wagon type
                ax.plot(times, inventories, label=f"{wagon_type}")

            # Set labels and title for the current plot
            ax.set_xlabel("Week - Day - Time")
            ax.set_ylabel("Number of Wagons")
            ax.set_title(f"Wagon Inventory Over Time - {terminal}")
            ax.legend()
            # Add major gridlines to both axes
            ax.grid(True, which='major', axis='both')

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, fontsize=8)
            plt.tight_layout()

            # Display the plot for the current terminal
            plt.show()

    def plot_results_time(self, configuration_type, timeline_inventory):
        # Iterate over each terminal to create a separate plot
        for terminal in self.terminals:
            fig, ax = plt.subplots(figsize=(14, 8))
            new_wagon_types = set(wagon_type for wagon_type, record_list in timeline_inventory[terminal].items())

            for wagon_type in new_wagon_types:
                # Extract time and inventory data for the current terminal and wagon type
                total_inventory = [(t, inventory) for _, t, inventory in timeline_inventory[terminal][wagon_type]]
                times, inventories = zip(*total_inventory)  # Unzip into separate lists

                # Plot the inventory over time for this terminal and wagon type
                ax.plot(times, inventories, label=f"{wagon_type}")

            # Set labels and title for the current plot
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel("Number of Wagons")
            ax.set_title(f"{configuration_type} Wagon Inventory Over Time - {terminal}")
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Display the plot for the current terminal
            plt.show()

    def plot_one_wagon(self, terminal, wagon_type):
        # Create a plot for the given terminal and wagon type
        fig, ax = plt.subplots(figsize=(14, 8))

        # Extract time and inventory data
        total_inventory = [(t, inventory) for _, t, inventory in self.timeline_inventory[terminal][wagon_type]]
        times, inventories = zip(*total_inventory)  # Unzip into separate lists

        # Plot the inventory over time
        ax.plot(times, inventories, label=f"{wagon_type}")

        # Add labels only when the inventory changes on the y-axis
        previous_inventory = None
        for i in range(len(times)):
            current_inventory = inventories[i]
            if current_inventory != previous_inventory:
                ax.text(times[i], current_inventory, f"{current_inventory}", fontsize=8, ha='right', va='bottom')
                previous_inventory = current_inventory

        # Set labels and title
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Number of Wagons")
        ax.set_title(f"Wagon Inventory Over Time - {terminal}")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Display the plot
        plt.show()

    def write_or_append(self, df, file_path):
        # Check if file exists, then append or write new data
        # mode = 'a' if os.path.exists(file_path) else 'w'
        # header = not os.path.exists(file_path)
        # df.to_csv(file_path, mode='w', header=header, index=False)
        # Always overwrite the file with new data
        df.to_csv(file_path, mode='w', header=True, index=False)

    def write_to_excel(self, df, file_path, sheet_name):
        # Write the DataFrame to a specific sheet
        """Explanation
            file_path: The path to your Excel file.
            sheet_name: The name of the sheet where the DataFrame will be written. If the sheet already exists, it will be overwritten.
            mode='a': Opens the file in append mode so you can add sheets to an existing file. If the file doesn't exist, it will be created.
            index=False: Prevents the DataFrame’s index from being written in the Excel sheet."""
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Load the workbook and hide the sheet
        # wb = load_workbook(file_path)
        # ws = wb[sheet_name]
        # ws.sheet_state = 'hidden'  # Options are 'visible', 'hidden', or 'veryHidden'
        # wb.save(file_path)

    # Simulate for multiple weeks

    def run_one_timetable(self):
        # my_timetable = TimeTable(self.access_db, self.inventory_output, self.adjustments_output, self.num_weeks)

        # graph = TrainGraph(my_timetable.train_services)
        # graph.create_graph()
        # graph.create_one_graph('2MP5')

        self.balance_inventory()
        self.convert_to_data_frame()
    # my_timetable.plot_results()
    # my_timetable.plot_results_time('Current Configuration', my_timetable.timeline_inventory)
    # my_simulation.plot_one_wagon('MFT', 'RQSY')
    # my_simulation.plot_results_time('Bowtie Configuration', my_simulation.bowtie_timeline_inventory)
