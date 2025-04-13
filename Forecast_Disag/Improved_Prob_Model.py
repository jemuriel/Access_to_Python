import itertools

import pandas as pd
import numpy as np

from Access_DB.ConnectDB import ReadAccessDB


class Probabilistic_Model:
    def __init__(self, historical_file, forecast_file, box_types_file, output_file):
        self.history_df = historical_file
        self.initial_forecast_df = forecast_file
        self.box_types = box_types_file
        self.output_file = output_file

        self.aggregated_forecast = None
        self.history_df_teus = None

        # self.train_data = self._process_access_db()

        self.train_names = None

        self.time_table_forecast = None
        self.container_probabilities = None
        self.teu_stats = None

        self.disaggregated_forecast = None

    def _process_access_db(self):
        # Read train names form the access DB
        db_reader = ReadAccessDB()
        trains_data = db_reader.read_table("tblTotalServicesByTrain")
        trains_data = trains_data[['Train', 'Departing Location', 'Departure Day of Week', 'Arriving Location']]
        trains_data = trains_data.drop_duplicates()
        trains_data = trains_data.dropna()
        trains_data.columns = ['TRAIN_NUM', 'ORIGIN', 'DOW', 'DESTINATION']

        trains_data.sort_values(by=['TRAIN_NUM'])
        trains_data['TRAIN_NUM'] = trains_data['TRAIN_NUM'].str.upper()

        # Assuming 'column_name' is the column with duplicate names
        trains_data['duplicate_count'] = trains_data.groupby('TRAIN_NUM').cumcount() + 1

        # Concatenate the name with the count for duplicates (only for duplicates)
        trains_data['TRAIN_NUM'] = trains_data.apply(
            lambda x: f"{x['TRAIN_NUM']}-L{x['duplicate_count']}",
            axis=1)

        # Drop the helper column used for counting duplicates
        trains_data = trains_data.drop('duplicate_count', axis=1)

        return trains_data

    def _process_forecast_data(self):
        # Convert DATE column to datetime format
        self.initial_forecast_df['DATE'] = pd.to_datetime(self.initial_forecast_df['DATE'], dayfirst=True,
                                                          errors='coerce')
        # self.forecast_df = self.forecast_df[self.forecast_df['DATE'].dt.month == self.month]
        self.corridors = self.initial_forecast_df['OD_PAIR'].unique()
        self.initial_forecast_df['MONTH'] = self.initial_forecast_df['DATE'].dt.month

    def _aggregate_forecast_by_train(self):

        corridor_train_subtotals = (self.history_df.groupby(['OD_PAIR', 'TRAIN_NUM', 'MONTH'])['SUBTOTAL_TEUS'].sum().
                                    reset_index(name='TEUS_CORRIDOR_TRAIN'))

        corridor_subtotals = self.history_df.groupby(['OD_PAIR', 'MONTH'])[
            'SUBTOTAL_TEUS'].sum().reset_index(name='TEUS_CORRIDOR')

        corridor_train_subtotals = corridor_train_subtotals.merge(corridor_subtotals, on=['OD_PAIR', 'MONTH'],
                                                                  how='left')
        corridor_train_subtotals['TRAIN_WEIGHT'] = (corridor_train_subtotals['TEUS_CORRIDOR_TRAIN'] /
                                                    corridor_train_subtotals['TEUS_CORRIDOR'])

        corridor_train_subtotals = corridor_train_subtotals.merge(self.initial_forecast_df, on=['OD_PAIR', 'MONTH'],
                                                                  how='left')

        # MAJOR CHANGE 'FORECASTED_VALUE' FOR 'NUM_BOXES'
        corridor_train_subtotals['NEW_FORECAST'] = (corridor_train_subtotals['NUM_BOXES'] *
                                                    corridor_train_subtotals['TRAIN_WEIGHT'])

        corridor_train_subtotals = corridor_train_subtotals.groupby(['TRAIN_NUM', 'MONTH', 'OD_PAIR']).agg(
            FORECASTED_VALUE=('NEW_FORECAST', 'sum'),
            DATE=('DATE', 'first')).reset_index()

        self.aggregated_forecast = corridor_train_subtotals

    def _process_historical_data(self):
        # Select the relevant columns and create a copy - FOR THE 2 YEARS FILE
        # self.history_df = self.history_df.iloc[:, [0, 4, 6, 7, 13, 14]].copy()

        #  --------- Filter the DataFrame to include rows where 'Actual.Origin.Train.No' contains "MB" or "BM" --------------------
        self.history_df = self.history_df[self.history_df['Actual.Origin.Train.No'].str.contains("MB|BM", na=False)]
        # -----------------------------------------------------------------------------------------------------------------------

        # FOR THE 5 YEARS FILE
        self.history_df = self.history_df.iloc[:, [0, 2, 4, 7, 13, 14]].copy()

        # Rename the columns
        self.history_df.columns = ['DESTINATION', 'DATE', 'ORIGIN', 'TRAIN_NUM', 'BOX', 'NUM_TEUS']

        self.history_df = self.history_df.dropna()
        self.train_names = self.history_df['TRAIN_NUM'].unique()

        # Convert DATE column to datetime format
        self.history_df['DATE'] = pd.to_datetime(self.history_df['DATE'], dayfirst=True, errors='coerce')

        # Check if DATE conversion was successful
        if self.history_df['DATE'].isnull().all():
            raise ValueError("DATE column conversion failed. Please check the input data.")

        # Create a month column
        if 'DATE' in self.history_df.columns and pd.api.types.is_datetime64_any_dtype(self.history_df['DATE']):
            self.history_df.loc[:, 'MONTH'] = self.history_df['DATE'].dt.strftime('%m').astype(
                int)  # Use .loc to avoid warning
            # self.history_df.loc[:, 'DOW'] = self.history_df['DATE'].dt.strftime('%a')  # Use .loc to avoid warning
        else:
            raise ValueError("DATE column is not in datetime format. Please check the conversion.")

        # Match with the box types
        self.history_df = self.history_df.merge(self.box_types, on='BOX', how='left')

        # Remove rows with any NA values
        self.history_df = self.history_df.dropna()
        self.history_df = self.history_df.dropna()

        # Create flows
        self.history_df['OD_PAIR'] = self.history_df['ORIGIN'] + '-' + self.history_df['DESTINATION']

        # Filter corridors for only the ones in the forecast
        self.history_df = self.history_df[self.history_df['OD_PAIR'].isin(self.corridors)]

        # Filter trains in the unique train names
        # self.history_df = self.history_df[self.history_df['TRAIN_NUM'].isin(self.train_names)]

        # Change container type to char
        self.history_df['BOX_TYPE'] = 'C' + self.history_df['BOX_TYPE'].astype(int).astype(str)

        # Remove empty origins or destinations
        self.history_df = self.history_df[(self.history_df['ORIGIN'] != '') & (self.history_df['DESTINATION'] != '')]
        self.history_df = self.history_df.drop(['DESTINATION', 'ORIGIN', 'BOX', 'NAME'], axis=1)

        # Create a custom 'WEEK' column based on the year and week number
        self.history_df['WEEK'] = self.history_df['DATE'].dt.strftime('%U').astype(int)

        # Check if necessary columns are present
        required_columns = ['TRAIN_NUM', 'DATE', 'MONTH', 'WEEK', 'BOX_TYPE', 'OD_PAIR']
        missing_columns = [col for col in required_columns if col not in self.history_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_columns)}")

        self.history_df_teus = self.history_df.copy()

        # Get the subtotals for TEUS prediction
        self.history_df = (self.history_df.groupby(['TRAIN_NUM', 'DATE', 'MONTH', 'WEEK', 'BOX_TYPE', 'OD_PAIR'])
                           .agg(SUBTOTAL_BOX=('NUM_TEUS', 'size'), SUBTOTAL_TEUS=('NUM_TEUS', 'sum')).reset_index())

        # self.history_df.to_csv('3_historical_data_MBM_Transformed.csv', index=False)

    def filter_historical_data(self):
        # Filter trains that are in the forecast
        mask = self.aggregated_forecast['TRAIN_NUM'].unique()
        self.history_df = self.history_df[self.history_df['TRAIN_NUM'].isin(mask)]

    def _create_timetable_forecast(self):
        # Map days (1 = Sunday, ..., 7 = Saturday)
        dow_mapping = {1: 'Sun', 2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'}

        # Create a mapping
        train_list = [(dow_mapping[int(train_name[0])], train_name) for train_name in self.train_names]

        # date_range = pd.date_range(start='2025-01-01', end='2025-01-31')
        # start_date = self.modified_forecast['DATE'].min()
        # end_date = self.modified_forecast['DATE'].max()
        # date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        year = self.aggregated_forecast['DATE'].dt.year.iloc[0]
        date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')

        # # Step 1: Get the number of days in the specified month
        # _, num_days = calendar.monthrange(self.year, self.month)
        #
        # # Step 2: Generate the date range for every day in the month
        # date_range = pd.date_range(start=f'{self.year}-{self.month:02d}-01',
        #                            end=f'{self.year}-{self.month:02d}-{num_days}', freq='D')

        timetable = pd.DataFrame({
            'DATE': date_range,
            'MONTH': [d.month for d in date_range],
            'WEEK': [d.strftime('%U') for d in date_range],
            'DOW': [d.strftime('%a') for d in date_range]
        })

        # Convert TRAIN dictionary to a DataFrame
        train_df = pd.DataFrame(train_list, columns=['DOW', 'TRAIN_NUM'])

        # Merge DATA_RANGE with train_df on DOW
        # This will repeat the DATE, MONTH, and DOW for each corresponding TRAIN_NAME
        new_df = timetable.merge(train_df, on='DOW', how='left')

        # Filter out rows where TRAIN_NUM is still None (i.e., no train assigned)
        new_df = new_df[new_df['TRAIN_NUM'].notnull()]

        # Transform week to number
        new_df['WEEK'] = new_df['WEEK'].astype(int)

        self.time_table_forecast = new_df

    def _calc_probabilities(self):
        # Get all the possible combinations between box type and train number
        week_unique = [f'{i:2}' for i in range(53)]  # Weeks from 00 to 52
        trains_unique = self.train_names  # List of unique train numbers

        # Count occurrences of container type in train/date combinations
        counts = self.history_df[['DATE', 'TRAIN_NUM', 'OD_PAIR', 'WEEK', 'BOX_TYPE']].drop_duplicates()
        all_counts = counts.groupby(['TRAIN_NUM', 'OD_PAIR', 'WEEK', 'BOX_TYPE']).size().reset_index(name='COUNT')

        # Remove duplicates based on the columns of interest
        unique_combinations = self.history_df[['TRAIN_NUM', 'OD_PAIR', 'DATE', 'WEEK']].drop_duplicates()

        # Group by and calculate total combinations for each train/week/DOW
        total_combinations = unique_combinations.groupby(['TRAIN_NUM', 'OD_PAIR', 'WEEK']).size().reset_index(
            name='TOTAL_COMBOS')

        # Merge counts with total combinations to calculate probability
        container_prob = pd.merge(all_counts, total_combinations, on=['TRAIN_NUM', 'OD_PAIR', 'WEEK'], how='left')
        container_prob['PRESENT_PROB'] = container_prob['COUNT'] / container_prob['TOTAL_COMBOS']

        # Create all possible combinations of TRAIN_NUM and WEEK
        combinations = list(itertools.product(trains_unique, week_unique))
        combinations_df = pd.DataFrame(combinations, columns=['TRAIN_NUM', 'WEEK'])

        # Transform week to number
        combinations_df['WEEK'] = combinations_df['WEEK'].astype(int)

        # Merge with container probabilities on TRAIN_NUM and WEEK
        combinations_df = combinations_df.merge(
            container_prob[['TRAIN_NUM', 'OD_PAIR', 'WEEK', 'BOX_TYPE', 'PRESENT_PROB']],
            on=['TRAIN_NUM', 'WEEK'], how='left')

        combinations_df = combinations_df.sort_values(by='TRAIN_NUM')

        # Function to replicate rows from the previous week if values are NaN
        def _replicate_empty_rows_from_next_week(df):
            # Sort DataFrame by TRAIN_NUM and WEEK to ensure proper order
            df.sort_values(by=['TRAIN_NUM', 'WEEK'], inplace=True)

            # Initialize a list to store new rows with fixed weeks
            new_fixed_rows = []

            # Get unique train names
            train_names_od_pair = df[['TRAIN_NUM', 'OD_PAIR']].drop_duplicates()

            # Loop through each TRAIN_NUM to process it independently
            for index, row in train_names_od_pair.iterrows():
                # Filter data for the specific train
                train_df = df[(df['TRAIN_NUM'] == row['TRAIN_NUM']) & (df['OD_PAIR'] == row['OD_PAIR'])]

                # Identify the reference week: all rows from the first week without NaN values in DOW, BOX_TYPE, PRESENT_PROB
                reference_week_df = train_df.dropna(subset=['BOX_TYPE', 'OD_PAIR', 'PRESENT_PROB'])

                # If no valid reference week exists, skip this train
                if reference_week_df.empty:
                    continue

                # Extract the earliest reference week with no NaN values
                first_valid_week = reference_week_df['WEEK'].min()
                reference_week_rows = reference_week_df[reference_week_df['WEEK'] == first_valid_week]

                # Loop through the train weeks to identify which ones need to be fixed (contain NaN values)
                for week_number in train_df['WEEK'].unique():
                    week_df = train_df[train_df['WEEK'] == week_number]

                    # If the current week has NaN values, replicate the reference week rows
                    if week_df.isna().any(axis=None):
                        # Copy the reference week's rows and change the 'WEEK' value to the current week
                        new_week_rows = reference_week_rows.copy()
                        new_week_rows['WEEK'] = week_number

                        # Append these new rows to the list
                        new_fixed_rows.append(new_week_rows)

            # Concatenate all new fixed rows with the original DataFrame
            if new_fixed_rows:
                # Concatenate original DataFrame and the new fixed rows
                df = pd.concat([df] + new_fixed_rows, ignore_index=True)

            # Drop rows that originally had NaN values in DOW, BOX_TYPE, or PRESENT_PROB
            df.dropna(subset=['BOX_TYPE', 'PRESENT_PROB'], inplace=True)

            # Reset the index
            df.reset_index(drop=True, inplace=True)

            return df

        # Apply replication to handle missing rows
        combinations_df = _replicate_empty_rows_from_next_week(combinations_df)

        # Drop remaining rows where DOW is still NaN (if any, after replication) - for WEEK = 0
        # combinations_df = combinations_df.dropna(subset=['DOW']).reset_index(drop=True)

        # Sort DataFrame by TRAIN_NUM and WEEK to ensure proper order
        combinations_df.sort_values(by=['TRAIN_NUM', 'WEEK'], inplace=True)

        # Store the result in the class attribute
        self.container_probabilities = combinations_df

    def _calc_avg_teu_weight(self, gamma=0.9):
        # Convert the 'DATE' column to datetime if not already
        self.history_df['DATE'] = pd.to_datetime(self.history_df['DATE'])

        # Extract the year from the 'DATE' column
        self.history_df['YEAR'] = self.history_df['DATE'].dt.year

        # Sort the dataframe by TRAIN_NUM, BOX_TYPE, and YEAR in descending order
        self.history_df = self.history_df.sort_values(by=['TRAIN_NUM','OD_PAIR', 'BOX_TYPE', 'YEAR'],
                                                      ascending=[True, True, True, False])

        # Create a new column 'YEAR_RANK' to assign the same offset to rows with the same YEAR
        # This ranks each unique year within each group of TRAIN_NUM and BOX_TYPE, starting from 0 for the most recent year
        self.history_df['YEAR_OFFSET'] = self.history_df.groupby(['TRAIN_NUM', 'OD_PAIR', 'BOX_TYPE'])['YEAR'].rank(
            method='dense', ascending=False).astype(int) - 1

        # Calculate the YEAR_WEIGHT using the exponential discounting based on the offset
        self.history_df['YEAR_WEIGHT'] = gamma ** self.history_df['YEAR_OFFSET']

        # Now proceed with the aggregation and weight calculation as usual
        teu_stats = self.history_df.groupby(['TRAIN_NUM', 'OD_PAIR', 'MONTH', 'WEEK', 'BOX_TYPE'], as_index=False).agg(
            TOTAL_TEUS_BOX=('SUBTOTAL_TEUS', 'sum'),
            YEAR_WEIGHT=('YEAR_WEIGHT', 'mean')  # Average weight across the group
        )

        # Multiply the TEUs by the YEAR_WEIGHT to adjust totals
        teu_stats['TOTAL_TEUS_BOX'] = teu_stats['TOTAL_TEUS_BOX'] * teu_stats['YEAR_WEIGHT']

        # 1. Calculate monthly totals by TRAIN_NUM, MONTH
        month_totals_by_train = teu_stats.groupby(['TRAIN_NUM', 'OD_PAIR', 'MONTH'])['TOTAL_TEUS_BOX'].sum().reset_index(
            name='TOTAL_TEUS_MONTH')

        teu_stats = teu_stats.merge(month_totals_by_train, on=['TRAIN_NUM', 'OD_PAIR', 'MONTH'], how='left')

        # 2. Calculate weekly totals by TRAIN_NUM, MONTH, and WEEK to normalize across weeks
        week_totals_by_train = (teu_stats.groupby(['TRAIN_NUM', 'OD_PAIR', 'MONTH', 'WEEK'])['TOTAL_TEUS_BOX'].sum().
        reset_index(name='TOTAL_TEUS_WEEK'))

        teu_stats = teu_stats.merge(week_totals_by_train, on=['TRAIN_NUM', 'OD_PAIR', 'MONTH', 'WEEK'], how='left')

        # 3. Calculate the weight of each week over the month total
        teu_stats['WEEK_WEIGHT'] = teu_stats['TOTAL_TEUS_WEEK'] / teu_stats['TOTAL_TEUS_MONTH']

        # 4. Calculate the weight of each box on the week total
        teu_stats['BOX_INTRA_WEEK_WEIGHT'] = teu_stats['TOTAL_TEUS_BOX'] / teu_stats['TOTAL_TEUS_WEEK']

        # Save the TEU stats with calculated weights for future use
        self.teu_stats = teu_stats

    def _create_forecast(self):

        def _prepare_disaggregated_forecast():
            """Merge timetable, container probabilities, forecast, and TEU stats into one DataFrame."""
            disag_forecast = pd.merge(
                self.time_table_forecast,
                self.container_probabilities[self.container_probabilities['PRESENT_PROB'] > 0]
                [['TRAIN_NUM', 'OD_PAIR', 'WEEK', 'BOX_TYPE']],
                on=['TRAIN_NUM', 'WEEK'], how='left'
            )

            disag_forecast = disag_forecast.merge(
                self.aggregated_forecast[['FORECASTED_VALUE', 'TRAIN_NUM', 'OD_PAIR', 'MONTH']],
                on=['TRAIN_NUM', 'OD_PAIR', 'MONTH'], how='left'
            )

            disag_forecast = disag_forecast.merge(
                self.teu_stats[
                    ['TRAIN_NUM', 'OD_PAIR', 'MONTH', 'WEEK', 'BOX_TYPE', 'WEEK_WEIGHT', 'BOX_INTRA_WEEK_WEIGHT']],
                on=['TRAIN_NUM', 'OD_PAIR', 'MONTH', 'WEEK', 'BOX_TYPE'], how='left'
            )

            return disag_forecast

        def _adjust_weekly_probabilities(disag_forecast):
            """Ensure the weekly probability for each train sums to 1."""
            # Step 1: Drop duplicates to get unique weekly weights
            unique_weights = disag_forecast.drop_duplicates(subset=['TRAIN_NUM', 'OD_PAIR', 'MONTH', 'WEEK_WEIGHT'])

            # Step 2: Calculate total weights and merge them back
            total_weights = unique_weights.groupby(['TRAIN_NUM', 'OD_PAIR', 'MONTH'])['WEEK_WEIGHT'].sum().reset_index(
                name='TOTAL_WEEK_WEIGHT')

            disag_forecast = disag_forecast.merge(total_weights, on=['TRAIN_NUM', 'OD_PAIR', 'MONTH'], how='left')

            # Step 3: Calculate and adjust remainder probability
            disag_forecast['REMAINDER_PROBABILITY'] = 1 - disag_forecast['TOTAL_WEEK_WEIGHT']

            # Step 4: Adjust WEEK_WEIGHT by distributing the remainder probability equally across weeks
            max_week = disag_forecast.groupby(['TRAIN_NUM', 'MONTH'])['WEEK'].max().reset_index(name='MAX_WEEK')
            disag_forecast = disag_forecast.merge(max_week, on=['TRAIN_NUM', 'MONTH'], how='left')
            disag_forecast['WEEK_WEIGHT'] += disag_forecast['REMAINDER_PROBABILITY'] / disag_forecast['MAX_WEEK']

            return disag_forecast

        def _calculate_teus(disag_forecast):
            """Calculate forecasted TEUs using WEEK_WEIGHT, BOX_INTRA_WEEK_WEIGHT, and FORECASTED_VALUE."""
            return (disag_forecast['WEEK_WEIGHT'] *
                    disag_forecast['BOX_INTRA_WEEK_WEIGHT'] *
                    disag_forecast['FORECASTED_VALUE'])

        def _calculate_number_of_boxes(disag_forecast):
            """Calculate the number of boxes based on forecasted TEUs and historical TEUs per box."""
            mean_teus_per_box = (self.history_df_teus.groupby(['TRAIN_NUM', 'OD_PAIR', 'BOX_TYPE'])['NUM_TEUS']
                                 .mean().reset_index(name='MEAN_NUM_TEUS_PER_BOX'))

            # Merge mean TEUs per box into disag_forecast
            disag_forecast = pd.merge(disag_forecast, mean_teus_per_box, on=['TRAIN_NUM', 'OD_PAIR', 'BOX_TYPE'],
                                      how='left')

            # Calculate number of boxes and round to the closest upper integer for each row
            disag_forecast['NUM_BOXES'] = np.ceil(
                disag_forecast['CALCULATED_TEUS'] / disag_forecast['MEAN_NUM_TEUS_PER_BOX'])

            not_nan_forecast = disag_forecast[disag_forecast['NUM_BOXES'].notna()]

            # Convert 'NUM_BOXES' to integer type for practical use
            not_nan_forecast['NUM_BOXES'] = not_nan_forecast['NUM_BOXES'].astype(int)

            return not_nan_forecast

        def _finalize_forecast(disag_forecast):
            """Clean up columns and save the final disaggregated forecast."""
            disag_forecast.drop(['REMAINDER_PROBABILITY', 'MAX_WEEK', 'TOTAL_WEEK_WEIGHT'], axis=1,
                                inplace=True)

            # disag_forecast.to_csv(self.output_file, index=False)

            return disag_forecast

        # Step 1: Prepare the disaggregated forecast by merging necessary data
        disag_forecast = _prepare_disaggregated_forecast()

        # Step 2: Ensure that the total weekly probability per month sums to 1
        disag_forecast = _adjust_weekly_probabilities(disag_forecast)

        # Step 3: Calculate the forecasted TEUs based on weights and forecasted values
        disag_forecast['CALCULATED_TEUS'] = _calculate_teus(disag_forecast)

        # Step 4: Calculate the number of boxes based on TEUs and average TEUs per box type
        disag_forecast = _calculate_number_of_boxes(disag_forecast)

        # Step 5: Clean up unnecessary columns and output the final forecast to a CSV file
        self.disaggregated_forecast = _finalize_forecast(disag_forecast)

    def run_complete_model(self):
        self._process_forecast_data()
        self._process_historical_data()
        self._aggregate_forecast_by_train()
        self.filter_historical_data()
        self._create_timetable_forecast()
        self._calc_probabilities()
        # self.calc_avg_teu_weight_with_decay()
        self._calc_avg_teu_weight()
        self._create_forecast()

        return self.disaggregated_forecast

    def run_process_data(self):
        self._process_forecast_data()
        self._process_historical_data()
        self._aggregate_forecast_by_train()
        self.filter_historical_data()
        self._create_timetable_forecast()
        self._calc_avg_teu_weight()
