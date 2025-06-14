from datetime import timedelta

import pandas as pd
import streamlit as st
import requests
from intervaltree import IntervalTree


# --- Supabase Config ---
url = "https://srgojfkwksdgtxtbabck.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNyZ29qZmt3a3NkZ3R4dGJhYmNrIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDM1NTQ3NSwiZXhwIjoyMDU5OTMxNDc1fQ.bt9uxyp3q1u-dcZWDzrjN4pdCDwbcuURvwIu2jNP3Is"

SUPABASE_URL = url
SUPABASE_KEY = key

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}

def fetch_table(table_name, chunk_size=1000):
    all_rows = []
    offset = 0
    while True:
        url = f"{SUPABASE_URL}/rest/v1/{table_name}?select=*"
        headers = {**HEADERS, "Range-Unit": "items", "Range": f"{offset}-{offset+chunk_size-1}"}
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        chunk = res.json()
        if not chunk:
            break
        all_rows.extend(chunk)
        if len(chunk) < chunk_size:
            break
        offset += chunk_size
    return pd.DataFrame(all_rows)


base_df = fetch_table("train_timetable_new")

corridor_mapping = {
    'MB': 'MB-BM', 'BM': 'MB-BM',
    'MP': 'MP-PM', 'PM': 'MP-PM',
    'SP': 'SP-PS', 'PS': 'SP-PS',
    'MS': 'MS-SM', 'SM': 'MS-SM',
    'SB': 'SB-BS', 'BS': 'SB-BS',
}
base_df['Corridor Group'] = base_df['Corridor'].map(corridor_mapping)
base_df['Datetime'] = pd.to_datetime(base_df['Datetime'], dayfirst=True)
base_df = base_df.sort_values(by='Datetime')
# base_df.to_csv(r"C:\Users\61432\Downloads\base_df.csv", index=False)
result = base_df.groupby('Train', group_keys=False).apply(lambda g: pd.concat([g.head(1), g.tail(1)])).reset_index()

def build_compatibility_matrix(train_meta_df, feature='WAGON_TYPE'):
    matrix = pd.DataFrame(0, index=train_meta_df.index, columns=train_meta_df.index)
    for i, row_i in train_meta_df.iterrows():
        for j, row_j in train_meta_df.iterrows():
            if row_i[feature] == row_j[feature]:
                matrix.loc[i, j] = 1
    return matrix

def yearly_timetable(weekly_df):
    # --- Step 1: Load Weekly Timetable ---
    # weekly_df = pd.read_csv("/mnt/data/train_timetable_new_rows (1).csv")
    weekly_df['Datetime'] = pd.to_datetime(weekly_df['Datetime'], dayfirst=True, errors='coerce')
    weekly_df = weekly_df.dropna(subset=["Datetime"]).copy()  # drop rows with invalid dates

    # --- Step 2: Setup Expansion for 2025 ---
    start_date = pd.Timestamp("2025-01-01")
    end_date = pd.Timestamp("2025-12-31")
    weeks_to_generate = pd.date_range(start=start_date, end=end_date, freq='W-SUN')  # all Sundays in 2025

    # Anchor the weekly pattern using the first train’s week
    base_week_start = weekly_df['Datetime'].min().normalize()

    # --- Step 3: Duplicate Weekly Schedule Across the Year ---
    expanded_dfs = []

    for i, week_start in enumerate(weeks_to_generate):
        week_shift = week_start - base_week_start
        df_copy = weekly_df.copy()
        df_copy['Datetime'] = df_copy['Datetime'] + week_shift
        df_copy['Train'] = df_copy['Train'].apply(lambda x: f"{i + 1}_{x}")  # unique train name for the week
        expanded_dfs.append(df_copy)

    # --- Step 4: Combine and Sort ---
    yearly_df = pd.concat(expanded_dfs, ignore_index=True)
    yearly_df = yearly_df.sort_values(by='Datetime')

    yearly_df.to_csv(r"C:\Users\61432\Downloads\yearly_df.csv", index=False)

    return yearly_df

def consist_optimiser_full_year(
    buffer_minutes: int = 720,
    max_cycle_days: int = 14
):
    df = yearly_timetable(result).copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.sort_values(by='Datetime')

    # Extract base train name
    df['BaseTrain'] = df['Train'].apply(lambda x: x.split('_', 1)[-1] if '_' in x else x)

    # Extract endpoints
    endpoints = (df.groupby('Train', group_keys=False)
                   .apply(lambda g: pd.concat([g.head(1), g.tail(1)]))
                   .reset_index(drop=True))

    services = endpoints[endpoints["Activity"] == "DEP"]
    arrivals = endpoints[endpoints["Activity"] == "ARR"]
    arr_lookup = arrivals.set_index("Train")["Terminal"].to_dict()
    arr_time_lookup = arrivals.set_index("Train")["Datetime"].to_dict()
    dep_lookup = services.set_index("Train")["Terminal"].to_dict()
    dep_time_lookup = services.set_index("Train")["Datetime"].to_dict()

    base_trains = df['BaseTrain'].unique()
    base_to_instances = df.groupby('BaseTrain')['Train'].unique().to_dict()

    consists = []
    consist_meta = []
    assigned_base = {}

    for base in sorted(base_trains):
        if base in assigned_base:
            continue

        candidate_trains = base_to_instances[base]
        assigned = False

        for cid, consist in enumerate(consists):
            last_train = consist[-1]
            last_arr_terminal = arr_lookup[last_train]
            last_arr_time = arr_time_lookup[last_train]

            first_train = candidate_trains[0]
            dep_terminal = dep_lookup[first_train]
            dep_time = dep_time_lookup[first_train]
            arr_time = arr_time_lookup[candidate_trains[-1]]

            if (last_arr_terminal == dep_terminal and
                last_arr_time + timedelta(minutes=buffer_minutes) <= dep_time):

                cycle_time = arr_time - consist_meta[cid]["start_time"]
                if cycle_time <= timedelta(days=max_cycle_days):
                    consist.extend(candidate_trains)
                    assigned_base[base] = cid
                    consist_meta[cid]["last_terminal"] = arr_lookup[candidate_trains[-1]]
                    assigned = True
                    break

        if not assigned:
            new_id = len(consists)
            consists.append(list(candidate_trains))
            assigned_base[base] = new_id
            consist_meta.append({
                "origin_terminal": dep_lookup[candidate_trains[0]],
                "start_time": dep_time_lookup[candidate_trains[0]],
                "last_terminal": arr_lookup[candidate_trains[-1]]
            })

    # Build result
    assignments = []
    for cid, trains in enumerate(consists, start=1):
        for t in trains:
            assignments.append((cid, t, dep_time_lookup[t], arr_time_lookup[t]))

    short_df = pd.DataFrame(assignments, columns=["Consist_ID", "Train", "Departure", "Arrival"])
    # long_train_df["Consist_ID"] = long_train_df["Train"].map(short_df.set_index("Train")["Consist_ID"])

    return short_df

# short_train_df = consist_optimiser_full_year()
# short_train_df.to_csv(r"C:\Users\61432\Downloads\short_train_df.csv", index=False)
# long_train_df.to_csv(r"C:\Users\61432\Downloads\long_train_df.csv", index=False)

# -----------------------------------
def consist_optimiser(long_train_df, buffer_minutes=720, max_cycle_days=14):
    df = long_train_df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], dayfirst=True)
    df = df.sort_values(by="Datetime")

    # Get endpoints for each train
    endpoints = df.groupby("Train", group_keys=False).apply(lambda g: pd.concat([g.head(1), g.tail(1)])).reset_index(drop=True)
    services = endpoints[endpoints["Activity"] == "DEP"]
    arrivals = endpoints[endpoints["Activity"] == "ARR"]

    arr_terminal = arrivals.set_index("Train")["Terminal"].to_dict()
    arr_time = arrivals.set_index("Train")["Datetime"].to_dict()
    dep_terminal = services.set_index("Train")["Terminal"].to_dict()
    dep_time = services.set_index("Train")["Datetime"].to_dict()

    trains = list(services["Train"].unique())
    used_trains = set()
    consists = []

    # Precompute feasible next train links
    successors = {train: [] for train in trains}
    for t1 in trains:
        for t2 in trains:
            if t1 == t2 or t2 in used_trains:
                continue
            if arr_terminal[t1] == dep_terminal[t2]:
                if arr_time[t1] + timedelta(minutes=buffer_minutes) <= dep_time[t2]:
                    successors[t1].append(t2)

    def build_consist_path(start_train):
        path = [start_train]
        visited = {start_train}
        origin = dep_terminal[start_train]
        cycle_start_time = dep_time[start_train]

        def dfs(current_train):
            for next_train in successors[current_train]:
                if next_train in visited or next_train in used_trains:
                    continue

                if arr_time[next_train] - cycle_start_time > timedelta(days=max_cycle_days):
                    continue

                path.append(next_train)
                visited.add(next_train)

                # If we return to origin, and the path is at least 2 trains long, stop
                if arr_terminal[next_train] == origin and len(path) > 1:
                    return True

                if dfs(next_train):
                    return True

                # backtrack
                path.pop()
                visited.remove(next_train)

            return False

        if dfs(start_train):
            return path
        return None

    for train in trains:
        if train in used_trains:
            continue
        path = build_consist_path(train)
        if path:
            consists.append(path)
            used_trains.update(path)

    # Assign consist IDs
    train_to_consist = {}
    assignments = []
    for cid, train_list in enumerate(consists, start=1):
        for t in train_list:
            train_to_consist[t] = cid
            assignments.append((cid, t, dep_time[t], arr_time[t]))

    short_df = pd.DataFrame(assignments, columns=["Consist_ID", "Train", "Departure", "Arrival"])
    long_train_df["Consist_ID"] = long_train_df["Train"].map(train_to_consist)

    return short_df, long_train_df

# Run the improved optimizer
short_train_df, long_train_df = consist_optimiser(result)
short_train_df.to_csv(r"C:\Users\61432\Downloads\short_train_df.csv", index=False)
long_train_df.to_csv(r"C:\Users\61432\Downloads\long_train_df.csv", index=False)

