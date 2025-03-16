import pandas as pd
import numpy as np
import random

from tqdm import tqdm

'''This class encapsulates the entire Q-learning process for calculating the weights and performing forecast disaggregation.'''


class TEUForecastQAgent:
    def __init__(self, historical_data, initial_weights,file_output, alpha=0.001, gamma=0.9, epsilon=0.1, decay_rate=0.995,
                 episodes=10):

        random.seed(42)

        # Initialize hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.decay_rate = decay_rate  # Epsilon decay rate
        self.total_episodes = episodes  # Number of learning episodes

        # Initialize historical and forecast data
        self.historical_data = historical_data
        self.output_file = file_output
        self.initial_weights = initial_weights

        # Define states, actions, and Q-table
        self.states = self._define_states()
        self.actions = ['increase_box_weight', 'decrease_box_weight']
        self.Q_table = self._initialize_q_table()
        self.learned_weights_df = None
        self.week_month_dict = self._get_month_from_week()

    def _get_month_from_week(self):
        week_month_df = self.historical_data[['MONTH', 'WEEK']].drop_duplicates()
        return dict(zip(week_month_df['WEEK'], week_month_df['MONTH']))

    def _define_states(self):
        """Create states as combinations of TRAIN_NUM, MONTH, WEEK, DOW, BOX_TYPE."""
        return list(self.initial_weights.groupby(['TRAIN_NUM', 'MONTH', 'WEEK', 'DOW', 'BOX_TYPE']).groups.keys())

    def _initialize_q_table(self):
        """Initialize the Q-table with the values from the Improved_Prob_Model"""
        Q_table = dict(zip(zip(self.initial_weights['TRAIN_NUM'],
                               self.initial_weights['MONTH'],
                               self.initial_weights['WEEK'],
                               self.initial_weights['DOW'],
                               self.initial_weights['BOX_TYPE']),
                           self.initial_weights['BOX_INTRA_WEEK_WEIGHT']))

        return Q_table

    def _calculate_reward(self, predicted_teus, actual_teus):
        reward = -abs(predicted_teus - actual_teus)
        # return np.clip(reward, 0, 1)  # Example of clamping reward
        return reward

    def _get_next_state(self, current_state):
        """Select the next week for the agent to explore."""
        train_num, month, week_num, dow, box_type = current_state

        # Return current state if the week number exceeds the maximum week
        if week_num >= 52:
            return current_state

        # Increment the week number
        week_num += 1

        # Continue finding the next valid state until we reach a valid state or max weeks
        increments = 0
        while week_num < 52 and increments < 2:
            # Get month corresponding to the new week number
            month = self.week_month_dict.get(week_num)

            # Break if no corresponding month found (to avoid KeyError)
            if month is None:
                print('Month is None in _get_next_state')
                break

            # Create the next state candidate
            next_state = (train_num, month, week_num, dow, box_type)

            # Check if next state is valid
            if next_state in self.states:
                return next_state

            # Increment week and update month
            week_num += 1
            increments += 1

        # If no valid next state is found, return the current state
        return current_state

    def _update_q_value(self, state, reward, next_state):
        """Update the Q-value using the Bellman equation."""
        # Find the maximum Q-value for the next state (which is a float)
        max_future_q = self.Q_table[next_state]  # Directly get the Q-value for the next state

        # Update the Q-value for the action taken in the current state
        current_q_value = self.Q_table[state]  # Get the current Q-value for the state

        # Apply the Q-learning update formula
        new_q_value = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_future_q)

        # Update the Q-value for the state
        self.Q_table[state] = new_q_value  # Update the Q-value for the state


    def _choose_action(self):
        """Choose an action using an epsilon-greedy strategy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            # Exploit: choose action with maximum reward
            return None  # Select the current value from q_table[state]

    def _get_actual_teus(self, state):
        """Get actual TEUs from historical data for a given state."""
        train_num, month, week, dow, box_type = state
        mask = ((self.historical_data['TRAIN_NUM'] == train_num) &
                (self.historical_data['MONTH'] == month) &
                (self.historical_data['WEEK'] == week) &
                (self.historical_data['DOW'] == dow) &
                (self.historical_data['BOX_TYPE'] == box_type))
        return self.historical_data.loc[mask, 'SUBTOTAL_TEUS'].mean()

    def _calculate_teus(self, box_weight, state):
        """Calculate predicted TEUs using the current weights."""
        train_num, month, week, _, _ = state
        calculated_value = self.historical_data[(self.historical_data['TRAIN_NUM'] == train_num) &
                                                (self.historical_data['MONTH'] == month)]['SUBTOTAL_TEUS'].sum()

        return box_weight * calculated_value

    def train(self):
        """Train the agent using Q-learning."""
        for episode in range(self.total_episodes):
            print()
            print(f'Episode {episode} --------------')
            for state in tqdm(self.states, desc="Processing States", unit="state"):
                # Choose an action based on the current state
                action = self._choose_action()

                # Get the current weights for the state if the action is None
                box_weight = self.Q_table[state]

                # Perform the action and adjust the weights
                if action == 'increase_box_weight':
                    box_weight = np.clip(box_weight + 0.05, 0, 1)
                elif action == 'decrease_box_weight':
                    box_weight = np.clip(box_weight - 0.05, 0, 1)

                # Calculate predicted TEUs
                predicted_teus = self._calculate_teus(box_weight, state)

                # Get the actual TEUs from historical data
                actual_teus = self._get_actual_teus(state)

                # Calculate the reward
                reward = self._calculate_reward(predicted_teus, actual_teus)

                # Get the next state
                next_state = self._get_next_state(state)

                # Update the Q-value
                self._update_q_value(state, reward, next_state)

            # Decay epsilon after each episode
            self.epsilon = self.epsilon * self.decay_rate

    def get_learned_weights(self):
        """Return the learned weights after training."""
        return self.Q_table

    def save_learned_weights(self):
        """Save the learned weights to a CSV file."""
        # Create a DataFrame from the dictionary items
        self.learned_weights_df = pd.DataFrame(list(self.Q_table.items()), columns=['Tuple', 'Value'])

        # Split the tuple into separate columns
        self.learned_weights_df[['TRAIN_NUM', 'MONTH', 'WEEK', 'DOW', 'BOX_TYPE']] = (
            pd.DataFrame(self.learned_weights_df['Tuple'].tolist(), index=self.learned_weights_df.index))

        # Drop the original tuple column if not needed
        self.learned_weights_df = self.learned_weights_df.drop(columns=['Tuple'])

        # Rename value column
        self.learned_weights_df.rename(columns={'Value': 'BOX_INTRA_WEEK_WEIGHT'}, inplace = True)

        def normalised_weights():
            # Calculate monthly weights
            train_monthly_weight = (self.learned_weights_df.groupby(['TRAIN_NUM', 'MONTH'])['BOX_INTRA_WEEK_WEIGHT'].sum().
                                    reset_index(name='MONTHLY_WEIGHT'))
            # Merge with learned weights
            self.learned_weights_df = self.learned_weights_df.merge(train_monthly_weight, on=['TRAIN_NUM', 'MONTH'], how='left')

            # Normalised the values
            self.learned_weights_df['NORMALISED_BOX_WEIGHTS'] = (self.learned_weights_df['BOX_INTRA_WEEK_WEIGHT'] /
                                                                 self.learned_weights_df['MONTHLY_WEIGHT'])

            self.learned_weights_df.drop(['MONTHLY_WEIGHT'], axis=1, inplace=True)

        normalised_weights()


        # Save to csv
        self.learned_weights_df.to_csv(self.output_file, index=False)

    def run_Q_lerning(self):

        # Train the Q-learning agent
        self.train()

        # Get the learned weights
        self.get_learned_weights()

        # Save the learned weights
        self.save_learned_weights()


'''This class uses the learned weights from the Q-learning agent to disaggregate the forecast.'''
class TEUForecastDisaggregator:
    def __init__(self, forecast_data, timetable_forecast, learned_weights, historical_teus, output_file):
        self.forecast_data = forecast_data
        self.learned_weights = learned_weights
        self.historical_teus = historical_teus
        self.time_table_forecast = timetable_forecast
        self.output_file = output_file

    def create_forecast(self):
        def RL_prepare_disaggregated_forecast():
            """Merge timetable, container probabilities, forecast, and TEU stats into one DataFrame."""
            disag_forecast = pd.merge(
                self.time_table_forecast,
                self.forecast_data[['FORECASTED_VALUE', 'TRAIN_NUM', 'MONTH']],
                on=['TRAIN_NUM', 'MONTH'], how='left'
            )

            disag_forecast = disag_forecast.merge(
                self.learned_weights, on=['TRAIN_NUM', 'MONTH', 'WEEK', 'DOW'], how='left'
            )

            return disag_forecast

        def RL_adjust_weekly_probabilities(disag_forecast):
            """Ensure the monthly probability for each train sums to 1"""
            # Step 1: Calculate total weights per month and merge them back
            monthly_weights = disag_forecast.groupby(['TRAIN_NUM', 'MONTH'])['NORMALISED_BOX_WEIGHTS'].sum().reset_index(
                name='TOTAL_MONTHlY_WEIGHT')

            disag_forecast = disag_forecast.merge(monthly_weights, on=['TRAIN_NUM', 'MONTH'], how='left')

            # Step 2: Calculate and adjust remainder probability
            disag_forecast['REMAINDER_PROBABILITY'] = 1 - disag_forecast['TOTAL_MONTHlY_WEIGHT']

            # Step 3: Adjust WEEK_WEIGHT by distributing the remainder probability across weeks
            num_rows_in_month = disag_forecast.groupby(['TRAIN_NUM', 'MONTH']).size().reset_index(name='NUM_ROWS')

            disag_forecast = disag_forecast.merge(num_rows_in_month, on=['TRAIN_NUM', 'MONTH'], how='left')
            disag_forecast['NORMALISED_BOX_WEIGHTS'] += disag_forecast['REMAINDER_PROBABILITY'] / disag_forecast['NUM_ROWS']

            disag_forecast.drop(['REMAINDER_PROBABILITY', 'NUM_ROWS', 'TOTAL_MONTHlY_WEIGHT'], axis=1, inplace=True)

            return disag_forecast

        def RL_calculate_teus(disag_forecast):
            """Calculate forecasted TEUs using WEEK_WEIGHT, BOX_INTRA_WEEK_WEIGHT, and FORECASTED_VALUE."""
            return (disag_forecast['NORMALISED_BOX_WEIGHTS'] *
                    disag_forecast['FORECASTED_VALUE'])

        def RL_calculate_number_of_boxes(disag_forecast):
            """Calculate the number of boxes based on forecasted TEUs and historical TEUs per box."""
            mean_teus_per_box = (self.historical_teus.groupby(['TRAIN_NUM', 'BOX_TYPE'])['NUM_TEUS']
                                 .mean().reset_index(name='MEAN_NUM_TEUS_PER_BOX'))

            disag_forecast = pd.merge(disag_forecast, mean_teus_per_box, on=['TRAIN_NUM', 'BOX_TYPE'], how='left')

            return round(disag_forecast['CALCULATED_TEUS'] / disag_forecast['MEAN_NUM_TEUS_PER_BOX'])

        def RL_finalize_forecast(disag_forecast):
            """Clean up columns and save the final disaggregated forecast."""
            disag_forecast.to_csv(self.output_file, index=False)

        # Step 1: Prepare the disaggregated forecast by merging necessary data
        disag_forecast = RL_prepare_disaggregated_forecast()

        # Step 2: Ensure that the total weekly probability per month sums to 1
        disag_forecast = RL_adjust_weekly_probabilities(disag_forecast)

        # Step 3: Calculate the forecasted TEUs based on weights and forecasted values
        disag_forecast['CALCULATED_TEUS'] = RL_calculate_teus(disag_forecast)

        # Step 4: Calculate the number of boxes based on TEUs and average TEUs per box type
        disag_forecast['NUM_BOXES'] = RL_calculate_number_of_boxes(disag_forecast)

        # Step 5: Clean up unnecessary columns and output the final forecast to a CSV file
        RL_finalize_forecast(disag_forecast)

        # Save the final disaggregated forecast for further use
        self.disaggregated_forecast = disag_forecast


