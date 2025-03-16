import pyodbc
import pandas as pd

#To read tables from the access DB
class ReadAccessDB:
    def __init__(self, db_path):
        # Initialize the database path
        self.db_path = db_path

        # Create the connection string for pyodbc
        self.connection_string = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={self.db_path};Mode=Read;"

        # Table mappings
        self.my_tables = {
            'trains': 'tblTrain',
            'locations': 'tblLocations',
            'power': 'tblOD_Power',
            'route': 'tblRoute',
            'train_times': 'tblTrainTimesNewSingleWeek',
            'DoW': 'tblDayOfWeek',
            'service_times': 'tblServicesNew',
            'product': 'tblProduct',
            'wagons_service': 'tblWagonsByService',
            'services': 'tblServicesNew',
            'wagons_class': 'tblWagonsByClass',
            'service_train': 'tblService_By_Train_All_110209',
            'wagon_plan': 'qryWagon_Plan'
        }

    def read_all_tables(self):
        """Reads all tables into a dictionary of DataFrames."""
        # Connect to the Access database
        conn = pyodbc.connect(self.connection_string)

        # Dictionary to store DataFrames
        dataframes = {}

        # Loop through tables to load data into DataFrames
        for table_name, the_table in self.my_tables.items():
            query = f"SELECT * FROM {the_table}"
            df = pd.read_sql(query, conn)
            dataframes[table_name] = df

        # Close the connection
        conn.close()

        return dataframes

    def read_table(self, table_name):
        """Reads a specific table into a DataFrame."""
        # Connect to the Access database
        conn = pyodbc.connect(self.connection_string)

        # Execute the query and load data into a DataFrame
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)

        # Close the connection
        conn.close()

        return df

    def read_multiple_tables(self, table_names): # pass a list of table names
        """Reads multiple tables specified in a list into a dictionary of DataFrames."""
        # Connect to the Access database
        conn = pyodbc.connect(self.connection_string)

        # Dictionary to store DataFrames for requested tables
        dataframes = {}

        # Loop through the requested tables
        for table_name in table_names:
            # Verify if the table name exists in the mappings
            if table_name in self.my_tables:
                query = f"SELECT * FROM {self.my_tables[table_name]}"
                df = pd.read_sql(query, conn)
                dataframes[table_name] = df
            else:
                print(f"Table '{table_name}' not found in database.")

        # Close the connection
        conn.close()

        return dataframes

    def extract_query(self, query_name):
        # Connect to the Access database
        conn = pyodbc.connect(self.connection_string)
        # Query the data (use the query or table feeding the report)
        query = f"SELECT * FROM {query_name}"  # Replace with your actual query/table name

        # Load data into pandas DataFrame
        df = pd.read_sql(query, conn)

        # Export to CSV
        # df.to_csv(r'C:\path_to_output\output.csv', index=False)

        # Close connection
        conn.close()

        return df


if __name__ == "__main__":
    db_reader = ReadAccessDB(r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning\Sample Train Plan.accdb")
    all_data = db_reader.read_all_tables()
    # trains_data = db_reader.read_table("tblTrain")
    all_data['train_times'].to_csv(r"C:\Users\61432\Downloads\train_plan.csv", index=False)
    print()


