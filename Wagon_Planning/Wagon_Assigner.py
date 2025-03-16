import pandas as pd


class WagonAssigner:
    def __init__(self, container_demand, the_leg, leg_id, train_number):
        self.train_number = train_number
        self.date = None
        self.leg_id = leg_id
        self.leg = the_leg
        self.demand_containers = self.__sort_containers(container_demand)
        self.wagon_configuration = the_leg.wagon_configuration
        self.remaining_capacity = self.__initialize_wagon_capacity()
        self.wagon_assignments = {wagon_type: [] for wagon_type in self.wagon_configuration.keys()}
        self.unassigned_containers = []  # Track containers that could not be assigned
        self.summary_data = []  # Store summary information for each train, leg, and date
        self.unassigned_data = []  # Store unassigned container information for each train, leg, and date
        self.extra_remaining_capacity = self.__initialize_wagon_capacity()
        self.extra_capacity_summary = []

    def __sort_containers(self, container_demand):
        # Extract the container type (e.g., '20', '40', '48') from the end of the BOX_TYPE string
        # container_demand['BOX_TYPE'] = container_demand['BOX_TYPE'].str.extract(r'(\d+)', expand=False)
        container_demand.loc[:, 'BOX_TYPE'] = container_demand['BOX_TYPE'].str.extract(r'(\d+)', expand=False)

        # container_demand['BOX_TYPE'] = container_demand['BOX_TYPE'].fillna('0').astype(int)
        # Sort by date, container type (descending for larger containers first), and number of containers
        container_demand = container_demand.sort_values(by=['DATE', 'BOX_TYPE', 'NUM_BOXES'],
                                                        ascending=[True, False, False])
        return container_demand

    def __initialize_wagon_capacity(self):
        # Initialize the remaining capacity of each wagon pack
        wagon_capacity = {}
        for wagon_type, wagon in self.wagon_configuration.items():
            # pack_dic = {f'Pack_{j+1}': [wagon.size] * int(wagon.num_platforms) for j in range(wagon.num_wagons)}
            pack_dic = {f'Pack_{j + 1}': [int(wagon_size) for wagon_size in wagon.configuration]
                        for j in range(wagon.num_wagons)}
            wagon_capacity[wagon_type] = pack_dic
        return wagon_capacity

    def __greedy_algorithm(self, container_demand):
        # Greedy assignment of containers to wagons in packs
        for idx, row in container_demand.iterrows():
            container_type = int(row['BOX_TYPE'])  # Get container type (length in feet)
            num_containers = row['NUM_BOXES']  # Get number of containers of this type

            for _ in range(num_containers):
                # Track the best pack and wagon index for the container (one that leaves the least remaining space)
                best_pack = None
                best_wagon_type = None
                best_wagon_index = None
                min_remaining_capacity = float('inf')

                for wagon_type, packs in self.remaining_capacity.items():
                    for pack_name, capacities in packs.items():
                        for idx, capacity in enumerate(capacities):
                            if container_type <= capacity:
                                # Find the wagon within the pack that minimizes the remaining capacity
                                if capacity - container_type < min_remaining_capacity:
                                    best_pack = pack_name
                                    best_wagon_type = wagon_type
                                    best_wagon_index = idx
                                    min_remaining_capacity = capacity - container_type

                # Assign the container if a suitable wagon was found
                if best_pack is not None and best_wagon_type is not None:
                    self.remaining_capacity[best_wagon_type][best_pack][best_wagon_index] -= container_type
                    self.wagon_assignments[best_wagon_type].append((self.date, container_type, best_pack, best_wagon_index))
                else:
                    # If no suitable wagon was found, add the container to unassigned list
                    # print(f"Demand exceeds capacity: Container {container_type} could not be assigned.")
                    self.unassigned_containers.append({'Container_Type': container_type,
                                                       'Train_Number': self.train_number,
                                                       'Leg_ID': self.leg_id,
                                                       'Leg_OD': f'{self.leg.leg_origin_terminal}-'
                                                                 f'{self.leg.leg_destination_terminal}',
                                                       'Date': self.date,
                                                       'Num_Containers': 1})

    def __allocate_unassigned(self, unassigned_this_date):
        extra_wagon_assignments = {wagon_type: [] for wagon_type in self.wagon_configuration.keys()}

        def __greedy_unassigned():
            # Greedy assignment of unassigned containers
            for container_dic in unassigned_this_date:
                container_type = container_dic['Container_Type']  # Get container type (length in feet)

                # Track the best pack and wagon index for the container (one that leaves the least remaining space)
                best_pack = None
                best_wagon_type = None
                best_wagon_index = None
                min_remaining_capacity = float('inf')

                for wagon_type, packs in self.extra_remaining_capacity.items():
                    for pack_name, capacities in packs.items():
                        for idx, capacity in enumerate(capacities):
                            if container_type <= capacity:
                                # Find the wagon within the pack that minimizes the remaining capacity
                                if capacity - container_type < min_remaining_capacity:
                                    best_pack = pack_name
                                    best_wagon_type = wagon_type
                                    best_wagon_index = idx
                                    min_remaining_capacity = capacity - container_type

                # Assign the container if a suitable wagon was found
                if best_pack is not None and best_wagon_type is not None:
                    self.extra_remaining_capacity[best_wagon_type][best_pack][best_wagon_index] -= container_type
                    extra_wagon_assignments[best_wagon_type].append((container_type, best_pack, best_wagon_index))
                    print('best pack found')
                else:
                    print('No wagon found for unassigned container')

        __greedy_unassigned()

        # Delete unused packs
        for wagon_type in list(self.extra_remaining_capacity.keys()):
            for pack_num in list(self.extra_remaining_capacity[wagon_type].keys()):
                unused_capacity = sum(wagon_cap for wagon_cap in self.extra_remaining_capacity[wagon_type][pack_num])
                initial_capacity = sum(
                    int(wagon_cap) for wagon_cap in self.wagon_configuration[wagon_type].configuration)
                if unused_capacity == initial_capacity:
                    del self.extra_remaining_capacity[wagon_type][pack_num]

            if len(self.extra_remaining_capacity[wagon_type]) == 0:
                del (self.extra_remaining_capacity[wagon_type])

    def print_assignments(self):
        # Print the assignments
        for wagon_type, assigned_containers in self.wagon_assignments.items():
            print(f"{wagon_type} contains containers:")
            for container, pack_name, platform_index in assigned_containers:
                print(f"  - Container {container} assigned to {pack_name}, platform {platform_index + 1} "
                      f"(remaining capacity: {self.remaining_capacity[wagon_type][pack_name][platform_index]})")
        if self.unassigned_containers:
            print("Unassigned containers due to insufficient capacity:")
            for container in self.unassigned_containers:
                print(f"  - Container {container['Container_Type']}")

    def run_assignment(self):
        # Iterate through each unique date and assign containers accordingly
        unique_dates = self.demand_containers['DATE'].unique()

        for date in unique_dates:
            # print(f'Processing date: {date}')
            self.date = date
            demand_date = self.demand_containers[self.demand_containers['DATE'] == date]
            self.__greedy_algorithm(demand_date)
            # self.print_assignments()
            self.calculate_train_occupancy(self.remaining_capacity, self.summary_data)  # for the current configuration
            # self.generate_summary()
            self.generate_unassigned_summary()

            # Re-optimise unassigned containers
            unassigned_on_date = [container for container in self.unassigned_containers if container['Date'] == date]
            self.__allocate_unassigned(unassigned_on_date)
            self.calculate_train_occupancy(self.extra_remaining_capacity,
                                           self.extra_capacity_summary)  # for the extra wagons
            self.generate_summary_extra_capacity()

            # Reinitialize train capacity after each date
            self.remaining_capacity = self.__initialize_wagon_capacity()
            self.extra_remaining_capacity = self.__initialize_wagon_capacity()

    def calculate_train_occupancy(self, capacity_type, summary_file):
        total_foot_utilised = 0
        total_foot_capacity = 0
        packs_used = 0
        total_packs = 0
        feet_to_meters = 0.3048
        unassigned_length = feet_to_meters * sum(
            container['Container_Type'] * container['Num_Containers'] for container in self.unassigned_containers
            if container['Date'] == self.date)

        for wagon_type, packs in capacity_type.items():
            for pack_name, capacities in packs.items():
                total_packs += 1
                wagon = self.wagon_configuration[wagon_type]
                pack_foot_capacity = sum(int(wagon_size) for wagon_size in wagon.configuration)
                pack_foot_utilised = pack_foot_capacity - sum(capacity for capacity in capacities)

                total_foot_capacity += pack_foot_capacity * feet_to_meters
                total_foot_utilised += pack_foot_utilised * feet_to_meters
                if pack_foot_utilised > 0:
                    packs_used += 1

        occupancy = (total_foot_utilised / total_foot_capacity) if total_foot_capacity > 0 else 0

        # Store data for summary
        summary_file.append({
            'Train_Number': self.train_number,
            'Leg_OD': f'{self.leg.leg_origin_terminal}-{self.leg.leg_destination_terminal}',
            'Leg_ID': self.leg_id,
            'Date': self.date,
            'Total_Capacity': total_foot_capacity,
            'Total_Utilised': total_foot_utilised,
            'Occupancy': occupancy,
            'Total_Packs': total_packs,
            'Packs_Used': packs_used,
            'Unassigned_Containers': len([container for container in self.unassigned_containers
                                          if container['Date'] == self.date]),
            'Unassigned_Length': unassigned_length
        })

    def generate_summary(self):
        # Generate summary dataframe
        summary_df = pd.DataFrame(self.summary_data)
        # print(summary_df)
        self.summary_df = summary_df

    def generate_summary_extra_capacity(self):
        # Generate summary dataframe
        extra_summary = pd.DataFrame(self.extra_capacity_summary)
        # print(summary_df)
        self.extra_summary = extra_summary

    def generate_unassigned_summary(self):
        # Generate unassigned containers dataframe
        if self.unassigned_containers:
            unassigned_df = pd.DataFrame(self.unassigned_containers)
            # print(unassigned_df)
            self.unassigned_df = unassigned_df
        else:
            self.unassigned_df = pd.DataFrame(columns=['Container_Type', 'Train_Number', 'Leg_ID', 'Leg_OD', 'Date',
                                                       'Num_Containers'])

    def save_summaries(self):
        # Save both summary and unassigned containers dataframes to CSV
        self.summary_df.to_csv('train_summary.csv', index=False)
        self.extra_summary.to_csv('extra_train_summary.csv', index=False)
        self.unassigned_df.to_csv('unassigned_containers.csv', index=False)

    @staticmethod
    def run_yearly_assignment(disag_forecast, train_timetable, output_summary, wagon_assignments_file, unassigned_boxes):
        grouped_forecast = disag_forecast.groupby('TRAIN_NUM')
        unique_trains_forecast = grouped_forecast['TRAIN_NUM'].unique()
        matched_train_services = {key: value for key, value in train_timetable.train_services_dic.items()
                                  if key in unique_trains_forecast}

        summary_data = []
        unassigned_data = []
        extra_summary = []
        wagon_assignments = {}

        for train_number, legs_dic in matched_train_services.items():
            print(f'Running assignments for train {train_number}', '=' * 50)
            for leg_id, the_leg in legs_dic.items():
                # print(f'Leg: {leg_id}', '*' * 10)
                container_demand = grouped_forecast.get_group(train_number)
                my_assigner = WagonAssigner(container_demand, the_leg, leg_id, train_number)
                my_assigner.run_assignment()
                summary_data.extend(my_assigner.summary_data)
                unassigned_data.extend(my_assigner.unassigned_containers)

                if train_number not in wagon_assignments:
                    wagon_assignments[train_number] = {}

                if the_leg not in wagon_assignments[train_number]:
                    wagon_assignments[train_number][the_leg] = {}

                wagon_assignments[train_number][the_leg] = my_assigner.wagon_assignments
                extra_summary.extend(my_assigner.extra_capacity_summary)


        # Save combined summary and unassigned data to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_summary, index=False)

        unassigned_df = pd.DataFrame(unassigned_data)
        unassigned_df = (unassigned_df.groupby(['Container_Type', 'Train_Number', 'Leg_ID', 'Date'])['Num_Containers'].
                         sum().reset_index())
        unassigned_df.to_csv(unassigned_boxes, index=False)

        # Flatten the structure into a list of rows
        wagon_assignments_rows = []
        for train_num, subdict in wagon_assignments.items():
            for train_leg, subsubdictionary in subdict.items():
                for wagon_type, tuples in subsubdictionary.items():
                    for t in tuples:
                        # Unpack the tuple into separate columns
                        wagon_assignments_rows.append((train_num, train_leg, wagon_type, *t))

        # Create DataFrame
        wagon_assignments_df = pd.DataFrame(wagon_assignments_rows, columns=["Train", "Train_leg", 'Wagon_type',
                                                                             'Date', "Container", "Pack", "idx"])
        wagon_assignments_df.to_csv(wagon_assignments_file, index=False)

        extra_summary = pd.DataFrame(extra_summary)
        extra_summary.to_csv('Extra_wagon_summary.csv', index=False)

