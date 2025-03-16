import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Wagon_Planning.Util_Classes import Terminal, Wagon
from Wagon_Planning.Util_Classes import Train_Leg


# THIS CLASS READS THE WAGON CONFIGURATION AND TRAIN TIMETABLE FROM CSV FILES THAT WERE MODIFIED BY THE USER
class NewTimeTable:
    def __init__(self, time_buckets, timetable_file, wagon_config_file, wagon_mapping, inventory_output,
                 adjustments_output, wagon_configuration_output, num_weeks):
        self.num_weeks = num_weeks
        self.inventory_output = inventory_output
        self.adjustments_output = adjustments_output
        self.time_buckets = time_buckets
        self.wagon_mapping = pd.read_csv(wagon_mapping)

        # self.wagon_pool = {}  # {Wagon_Class:Wagon}

        self.time_range = range(0, self.num_weeks * 7 * 24 * 60, self.time_buckets)  # 10-minute intervals for one week

        self.train_services_df = self.initialise_train_services(timetable_file)
        self.train_services_dic = self.create_train_services()  # {train_num:{leg_number:train_leg}}
        self.wagon_plan = pd.read_csv(wagon_config_file)

        self.wagon_plan = self.platform_mapping(self.wagon_plan)

        self.wagon_types = set(self.wagon_plan['WAGON_CLASS'])

        self.terminals = self.create_terminals()

        self.timeline_inventory = self.create_inventory_structure()
        self.adjustment_recorder = self.create_inventory_structure()
        self.wagon_configuration_file = wagon_configuration_output
        self.create_current_config()

    def initialise_train_services(self, timetable_file):
        train_services = pd.read_csv(timetable_file)
        train_services['DEP_TIME'] = pd.to_datetime(train_services['DEP_TIME'], dayfirst=True,
                                                    errors='coerce')
        train_services['ARR_TIME'] = pd.to_datetime(train_services['DEP_TIME'], dayfirst=True,
                                                    errors='coerce')
        train_services['DEP_TIME'] = train_services['DEP_TIME'].dt.time
        train_services['ARR_TIME'] = train_services['ARR_TIME'].dt.time

        return train_services

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

    def create_current_config(self):
        # Trains in train services that are not in wagon plan ----------------------------------------------------------
        temp_trains = self.train_services_df['TRAIN_NAME'].unique()
        temp_wagon = self.wagon_plan['TRAIN_NAME'].unique()
        missing_trains = np.setdiff1d(temp_trains, temp_wagon)

        grouped_rows = self.wagon_plan.groupby('TRAIN_NAME')

        # Iterate over each train service
        for train_number, leg_dictionary in self.train_services_dic.items():
            # print(train_number)
            if train_number in missing_trains:
                continue

            # Filter rows for the current train number
            relevant_rows = grouped_rows.get_group(train_number)

            # Make the subtotals by wagon type
            relevant_rows = (relevant_rows.groupby(["TRAIN_NAME", 'WAGON_CLASS', "LEG_NUM", "PLATFORM", "NUM_PLATFORMS",
                                                    "FAMILY", 'SIZE', 'ALL_EQUAL', 'CONFIGURATION'])["NUM_WAGONS"].sum().reset_index(name='NUM_WAGONS'))

            # Iterate over each row for the current train number
            for _, row in relevant_rows.iterrows():
                # print(row)
                wagon_class = row['WAGON_CLASS']
                wagon_size = row['SIZE']
                leg_num = row['LEG_NUM']
                num_wagons = row['NUM_WAGONS']
                platform = row['PLATFORM']
                num_platforms = row['NUM_PLATFORMS']
                family = row['FAMILY']
                all_equal = row['ALL_EQUAL']
                configuration = row['CONFIGURATION']

                leg_wagon_configuration = leg_dictionary[leg_num].wagon_configuration

                # Update the wagon configuration for the specific leg
                leg_wagon_configuration[wagon_class] = Wagon(wagon_class, wagon_size, num_wagons, platform,
                                                             num_platforms, family, all_equal, configuration)

        self.delete_trains_with_empty_legs()
        self.convert_train_services_dic_to_df()

    def convert_train_services_dic_to_df(self):
        # List to store transformed data
        train_services_list = []

        # Iterate through the train services dictionary
        for train_number, train_data in self.train_services_dic.items():
            for leg_id, leg_data in train_data.items():
                leg_origin_destination = f"{leg_data.leg_origin_terminal}-{leg_data.leg_destination_terminal}"

                # Extract platform details
                for platform_name, platform_info in leg_data.wagon_configuration.items():
                    train_services_list.append({
                        'TRAIN_NUMBER': train_number,
                        'LEG_ID': leg_id,
                        'LEG_OD': leg_origin_destination,
                        'MINUTE_ARRIVAL': leg_data.arr_min_num,
                        'ARRIVAL_DAY': leg_data.arrival_day,
                        'MINUTE_DEPARTURE': leg_data.dep_min_num,
                        'DEPARTURE_DAY': leg_data.departure_day,
                        'PLATFORM_NAME': platform_name,
                        'NUMBER_OF_PLATFORMS': platform_info.num_platforms,
                        'PLATFORM_SIZE': platform_info.size,
                        'NUMBER_OF_WAGONS': platform_info.num_wagons,
                        'WAGON_TYPE': platform_info.type
                    })

        # Convert to DataFrame and save to CSV
        train_df = pd.DataFrame(train_services_list)
        train_df.to_csv(self.wagon_configuration_file, index=False)

    def delete_trains_with_empty_legs(self):
        keys_to_remove = {
            train_number
            for train_number, train_leg_dic in self.train_services_dic.items()
            for leg_number, train_leg in train_leg_dic.items()
            if len(train_leg.wagon_configuration) == 0
        }

        for train_number in keys_to_remove:
            del self.train_services_dic[train_number]

    def create_inventory_structure(self):
        """Create an empty inventory structure for terminals and wagon types."""
        return {terminal: {wagon_type: [] for wagon_type in self.wagon_types} for terminal in
                self.terminals}

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
            if NewTimeTable.is_within_time_bucket(
                    train_leg.dep_min_num if adjustment < 0 else train_leg.arr_min_num, t,
                    NewTimeTable.min_to_weeknum(t), self.time_buckets):
                for wagon_type, wagon_number in wagon_config.items():
                    wagon_inventory[terminal][wagon_type] += adjustment * wagon_number

                    self.adjustment_recorder[terminal][wagon_type].append(
                        (NewTimeTable.min_to_weeknum(t), t, train_number, leg_number,
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
                        (NewTimeTable.min_to_weeknum(t), t, wagon_inventory[terminal][wagon_type]))

        NewTimeTable.remove_zero_values(self.timeline_inventory)

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
                        'Wagon_Type': wagon_type,
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
                        'Wagon_Type': wagon_type,
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
        #
        # inventory_df = self.platform_mapping(inventory_df)
        # adjustment_df = self.platform_mapping(adjustment_df)

        self.write_or_append(inventory_df, self.inventory_output)
        self.write_or_append(adjustment_df, self.adjustments_output)
        # self.write_to_excel(inventory_df, self.file_output, "inventory_balance")
        # self.write_to_excel(adjustment_df, self.file_output, "inventory_adjustments")

        # return (self.inventory_output, self.adjustments_output)

    def platform_mapping(self, df):
        # leave only intermodal
        self.wagon_mapping = self.wagon_mapping[self.wagon_mapping['UNIT'] == 'Intermodal']
        self.wagon_mapping.drop("UNIT", axis=1, inplace=True)
        df = df.merge(self.wagon_mapping, on='WAGON_CLASS', how='left')
        # df = df.fillna('No_mapping')
        df = df[df['PLATFORM'].notna()]
        # df = df.dropna(how='any')
        # df.drop('Wagon_Type', axis=1, inplace=True)

        return df

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
