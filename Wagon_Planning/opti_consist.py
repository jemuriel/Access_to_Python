from datetime import timedelta

import pandas as pd
import streamlit as st
import requests

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

def improved_consist_optimiser(df, buffer_minutes=30):
    df = df.sort_values("Datetime")
    services = df[df["Activity"] == "DEP"]
    arr_lookup = df[df["Activity"] == "ARR"].set_index("Train")["Terminal"].to_dict()
    arr_time_lookup = df[df["Activity"] == "ARR"].set_index("Train")["Datetime"].to_dict()

    trains = services["Train"].unique()
    compatibility_matrix = pd.DataFrame(1, index=trains, columns=trains)  # All compatible

    consists = []
    assignments = []
    buffer_time = timedelta(minutes=buffer_minutes)

    for idx, row in services.iterrows():
        candidate = None
        best_gap = None

        for consist_id in range(len(consists)):
            last_train = consists[consist_id][-1]
            last_arr_terminal = arr_lookup.get(last_train)
            last_arr_time = arr_time_lookup.get(last_train)

            if (compatibility_matrix.at[last_train, row["Train"]] == 1 and
                last_arr_terminal == row["Terminal"] and
                last_arr_time + buffer_time <= row["Datetime"]):
                gap = (row["Datetime"] - last_arr_time).total_seconds()
                if best_gap is None or gap < best_gap:
                    candidate = consist_id
                    best_gap = gap

        if candidate is not None:
            consists[candidate].append(row["Train"])
            assignments.append((candidate, row["Train"], row["Datetime"], arr_time_lookup.get(row["Train"])))
        else:
            new_id = len(consists)
            consists.append([row["Train"]])
            assignments.append((new_id, row["Train"], row["Datetime"], arr_time_lookup.get(row["Train"])))

    # Post-processing: Try to merge underused consists
    merged = True
    while merged:
        merged = False
        for i in range(len(consists)):
            if not consists[i]:
                continue
            last_train = consists[i][-1]
            last_arr_terminal = arr_lookup.get(last_train)
            last_arr_time = arr_time_lookup.get(last_train)
            for j in range(len(consists)):
                if i == j or not consists[j]:
                    continue
                next_train = consists[j][0]
                next_dep_terminal = services[services['Train'] == next_train]["Terminal"].values[0]
                next_dep_time = services[services['Train'] == next_train]["Datetime"].values[0]

                if (compatibility_matrix.at[last_train, next_train] == 1 and
                    last_arr_terminal == next_dep_terminal and
                    last_arr_time + buffer_time <= next_dep_time):

                    consists[i].extend(consists[j])
                    consists[j] = []
                    merged = True
                    break
            if merged:
                break

    # Rebuild assignment list
    assignments = []
    for cid, trains in enumerate(consists):
        for t in trains:
            dep = services[services['Train'] == t]["Datetime"].values[0]
            arr = arr_time_lookup.get(t)
            assignments.append((cid, t, dep, arr))

    return pd.DataFrame(assignments, columns=["Consist ID", "Train", "Departure", "Arrival"])


num_consists = improved_consist_optimiser(result)
num_consists.to_csv(r"C:\Users\61432\Downloads\opti_consist_improved.csv", index=False)
