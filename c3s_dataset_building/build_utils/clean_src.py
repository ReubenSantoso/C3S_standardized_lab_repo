"""
c3sdb/build_utils/clean_src.py

Reuben Santoso (reubens@uw.edu)

New Pushes Made:
1. Created unit test cases for all functions
2. Significant changes made to process entries logic handling
    - adding logic to groups with only one entry
    - refining logic for entries with more than two entries
3. Three new datasets added

Run: python3 -m c3sdb.build_utils.clean_src
"""

import sqlite3
import numpy as np
import os
import pandas as pd

from build_utils.src_data import _gen_id
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

INCLUDE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_include"
)

def remove_invalid_smiles(db_path, clean_db_path):
    """
    Removes invalid SMILES structures from a dataframe.
    An invalid SMILES is when an entry has more than 1 SMILES string (split by '.')
    or if the SMILES is null/None.
    """

    # Initialize clean database with schema if it doesn't exist
    if not os.path.exists(clean_db_path):
        print(f"Creating new clean database: {clean_db_path}")
        create_clean_db(clean_db_path)

    # Read data from the original database into a pandas DataFrame
    conn = sqlite3.connect(db_path)
    df_master = pd.read_sql_query(
        "SELECT * FROM master",
        conn
    )
    df_master_col = df_master.columns.tolist()

    df_mqn = pd.read_sql_query(
        "SELECT * FROM mqns",
        conn
    )
    df_mqn_col = df_mqn.columns.tolist()

    # Merge DataFrames by g_id (left join to keep all master entries)
    combined_df = pd.merge(df_master, df_mqn, on='g_id', how='left')

    combined_df = combined_df[combined_df['smi'].notnull()] #only keep entries with SMILE string
    combined_df['smi'] = combined_df['smi'].apply(lambda x: x.split('.'))
    combined_df = combined_df[combined_df['smi'].apply(lambda x: len(x) == 1)] #keep only entries with one SMILE string
    combined_df['smi'] = combined_df['smi'].apply(lambda x: x[0])  # Convert list back to string

    #push the combined_df to clean on their respective tables
    combined_df[df_master_col].to_sql('master', sqlite3.connect(clean_db_path), if_exists='append', index=False)
    combined_df[df_mqn_col].to_sql('mqns', sqlite3.connect(clean_db_path), if_exists='append', index=False)

    conn.close()

def calculate_rsd(values):
    """
    Calculates the relative standard deviation (RSD) for a list of values

    Parameters
    ----------
    values : ``list``
        list of values to calculate the RSD

    Returns
    -------
    ``float``
        RSD of the input values
    """
    return np.std(values) / np.mean(values) * 100

def remove_outliers_and_average(values):
    """
    Removes outliers from the list and averages the remaining values if their RSD is below 1%.
    This method will always receive input that has outliers, since called if RSD is bigger than 1%.

    Parameters
    ----------
    values : list
        List of NUMERICAL values from which to remove values.

    Returns
    -------
    float
        Average of the values if RSD < 1% after removing outliers, or best possible average.
    """
    print("HIT remove outliers")
    max_iterations = 10  # Set a limit for the number of iterations
    iteration = 0

    while calculate_rsd(values) > 1:
        print(f"In while loop, iteration {iteration}")
        
        if iteration >= max_iterations:
            print("Reached maximum iterations, returning best possible average.")
            break
        
        if len(values) <= 2:  # If there are too few values left, break the loop
            print("Too few values left, breaking out of the loop.")
            break
        
        # Calculate mean and standard deviation
        mean_val, std_dev  = np.mean(values), np.std(values)
        
        # Retain values that are within one standard deviation from the mean
        filtered_values = [v for v in values if abs(v - mean_val) <= std_dev]

        # If filtering doesn't change the list, stop to prevent infinite loop
        if len(filtered_values) == len(values):
            print("No change after filtering, breaking out of the loop.")
            break
        
        values = filtered_values
        iteration += 1

    return values

def process_entries(entries):
    """
    Processes entries to either average their CCS or leave them unchanged based on RSD and ccs_type

    Parameters
    ----------
    entries : ``list``
        list of entries containing CCS from GROUPED entries

    Returns
    -------
    ``list``
        list of processed entry dictionaries with updated CCS values
    """
    ccs_values = [e["ccs"] for e in entries]
    rsd = calculate_rsd(ccs_values)
    
    dt_entries = [e for e in entries if e["ccs_type"] == "DT"]
    dt_ccs_entries = [e["ccs"] for e in dt_entries]

    # logic for handling duplicate entries (exactly two entries in group)
    if len(ccs_values) == 2:
        print("processing two entries")
        if rsd <= 1:
            print("two entries: RSD less than 1")
            # if RSD < 1%, simply average the values
            averaged_value = round(np.mean(ccs_values), 4)
            # Return single entry with averaged CCS
            result_entry = entries[0].copy()
            result_entry["ccs"] = averaged_value
            return [result_entry]
        
        # if RSD > 1%, process based on ccs_type
        else:
            print("two entries: RSD more than one continue with more")
            # Handling cases with exactly two entries and DT considerations
            if len(dt_entries) == 1:
                # If exactly one entry is DT and RSD > 1%, keep only the DT measurement
                return dt_entries
            elif len(dt_entries) == 2:
                # If both are DT and RSD > 1%, keep both values
                return entries
            else:
                # If no entries are DT and RSD > 1%, keep both values
                return entries
        
    elif len(ccs_values) > 2:
        print("processing more than two entries")
        
        if rsd <= 1:  # If RSD < 1%, average all CCS values
            print("more than two entries: RSD less than 1")
            averaged_value = round(np.mean(ccs_values), 4)
            # Return single entry with averaged CCS
            result_entry = entries[0].copy()
            result_entry["ccs"] = averaged_value
            return [result_entry]
    
        else:  # If RSD > 1%, attempt to remove outliers and recheck RSD
            print("more than two entries: RSD more than one continue with more")
            if len(dt_entries) >= 1:
                print("processing DT entries ONLY")
                processed_ccs = remove_outliers_and_average(dt_ccs_entries)
                # Update DT entries with processed CCS values
                result_entries = []
                for i, dt_entry in enumerate(dt_entries):
                    if i < len(processed_ccs):
                        updated_entry = dt_entry.copy()
                        updated_entry["ccs"] = processed_ccs[i]
                        result_entries.append(updated_entry)
                return result_entries
            else:
                print("processing entries with no DT")
                processed_ccs = remove_outliers_and_average(ccs_values)
                # Update entries with processed CCS values
                result_entries = []
                for i, entry in enumerate(entries):
                    if i < len(processed_ccs):
                        updated_entry = entry.copy()
                        updated_entry["ccs"] = processed_ccs[i]
                        result_entries.append(updated_entry)
                return result_entries
            
    else:
        print("processing one entry directly return")
        return entries  # Return the single entry as-is


def create_clean_db(clean_db_path):
    """
    Creates the clean database using the schema files from the specified include path

    Parameters
    ----------
    clean_db_path : ``str``
        path to clean database, if it exists
    include_path : ``str``
        path to SQLite3 schema files
    """
    if os.path.exists(clean_db_path):
        os.remove(clean_db_path)

    con = sqlite3.connect(clean_db_path)
    cur = con.cursor()
    # point to correct SQLit3 schema scripts
    sql_scripts = [
        os.path.join(INCLUDE_PATH, "C3SDB_schema.sqlite3"),
        os.path.join(INCLUDE_PATH, "mqn_schema.sqlite3"),
        os.path.join(INCLUDE_PATH, "pred_CCS_schema.sqlite3"),
    ]
    for sql_script in sql_scripts:
        with open(sql_script, "r") as sql_file:
            cur.executescript(sql_file.read())
    con.commit()
    con.close()


def clean_database(db_path, clean_db_path):
    """
    Cleans and prepares a new database using the schema files

    Parameters
    ----------
    db_path : str
        path to original C3S database file
    clean_db_path : str
        path to clean database file
    """

    # INSERT_YOUR_CODE
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"The provided db_path does not exist: {db_path}")
    if db_path is None or db_path.strip() == "":
        raise ValueError("db_path is not provided or is empty.")

    # Initialize clean database with schema if it doesn't exist
    if not os.path.exists(clean_db_path):
        print(f"Creating new clean database: {clean_db_path}")
        create_clean_db(clean_db_path)

    # Read data from the original database into a pandas DataFrame
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT g_id, name, adduct, mass, z, mz, ccs, smi, chem_class_label, src_tag, ccs_type, ccs_method FROM master",
        conn
    )
    df_master_col = df.columns.tolist()

    df_mqn = pd.read_sql_query(
        "SELECT * FROM mqns",
        conn
    )
    df_mqn_col = df_mqn.columns.tolist()

    df = pd.merge(df, df_mqn, on='g_id', how='left')
    
    # Normalize the name to lowercase and round mz for grouping
    df['name'] = df['name'].str.lower()
    df['rounded_mz'] = df['mz'].round(0)
    df['ccs'] = df['ccs'].astype(int)
    
    # Group by name, adduct, and rounded mz
    grouped = df.groupby(['name', 'adduct', 'rounded_mz'])
    
    # Open connection once for the whole database
    clean_conn = sqlite3.connect(clean_db_path)

    # Initialize entry counter
    entry_count = 0  

    for group_key, group_df in grouped:
        print(f"🔴 Entries processed: {entry_count}/{len(grouped)} 🔴 ")            

        if len(group_df) > 1:
            print(f"✅ Processing group with key: {group_key} {group_df} and size: {len(group_df)}")
            
            print("converting into dictonary")
            group_entries = group_df.to_dict(orient='records')
            
            processed_entries = process_entries(group_entries)    
            
            print(f"🟠 Processed {len(processed_entries)} entries 🟠")
            
            for processed_entry in processed_entries:
                try:
                    # Create DataFrames by selecting only the columns that exist in each table
                    master_data = {col: processed_entry[col] for col in df_master_col if col in processed_entry}
                    mqn_data = {col: processed_entry[col] for col in df_mqn_col if col in processed_entry}
                    
                    entry_master = pd.DataFrame([master_data])
                    entry_mqn = pd.DataFrame([mqn_data])

                    # Insert the entry into the master table (append mode)
                    entry_master.to_sql('master', clean_conn, if_exists='append', index=False)
                    entry_mqn.to_sql('mqns', clean_conn, if_exists='append', index=False)

                    print(f"🟠 Inserted processed group with key: {group_key}")
                    entry_count += 1
                except Exception as e:
                    print(f"❌ Error processing group with key: {group_key}. Error: {e}")
                    print(f"Debug - processed_entry keys: {list(processed_entry.keys()) if isinstance(processed_entry, dict) else 'Not a dict'}")
                    print(f"Debug - master_data: {master_data if 'master_data' in locals() else 'Not created'}")
                    print(f"Debug - mqn_data: {mqn_data if 'mqn_data' in locals() else 'Not created'}")
                    continue

        elif len(group_df) == 1:
            try:
                # Prepare entry for a single-row group, drop 'rounded_mz'
                single_entry_df = group_df.drop(columns=['rounded_mz'])
                # Convert single row to dictionary
                single_entry = single_entry_df.iloc[0].to_dict()
                
                # Create DataFrames by selecting only the columns that exist in each table
                master_data = {col: single_entry[col] for col in df_master_col if col in single_entry}
                mqn_data = {col: single_entry[col] for col in df_mqn_col if col in single_entry}
                
                entry_master = pd.DataFrame([master_data])
                entry_mqn = pd.DataFrame([mqn_data])
                
                entry_master.to_sql('master', clean_conn, if_exists='append', index=False)
                entry_mqn.to_sql('mqns', clean_conn, if_exists='append', index=False)

                entry_count += 1
            except Exception as e:
                print(f"❌ Error processing group with key: {group_key}. Error: {e}")
                print(f"Debug - single_entry keys: {list(single_entry.keys()) if 'single_entry' in locals() and isinstance(single_entry, dict) else 'Not a dict'}")
                print(f"Debug - master_data: {master_data if 'master_data' in locals() else 'Not created'}")
                print(f"Debug - mqn_data: {mqn_data if 'mqn_data' in locals() else 'Not created'}")
                continue

        
    # Close connections after processing all groups
    conn.close()  # Close original database connection
    clean_conn.commit()
    clean_conn.close()
    
    print(f"Database cleaned and saved as {clean_db_path}, entries added {entry_count}")
    print(f"✅ Clean DB done")
    

if __name__ == "__main__":
    clean_database("C3S.db", "C3S_clean.db")
