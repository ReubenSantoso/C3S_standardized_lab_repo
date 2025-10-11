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

    # Read data from the original database into a pandas DataFrame
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT g_id, name, adduct, mass, z, mz, ccs, smi, chem_class_label, src_tag, ccs_type, ccs_method FROM master",
        conn
    )
    conn.close()

    df = df[df['smi'].notnull()] #only keep entries with SMILE string
    df['smi'] = df['smi'].apply(lambda x: x.split('.'))
    df = df[df['smi'].apply(lambda x: len(x) == 1)] #keep only entries with one SMILE string
    df['smi'] = df['smi'].apply(lambda x: x[0])  # Convert list back to string

    clean_conn = sqlite3.connect(clean_db_path)
    df.to_sql('master', clean_conn, if_exists='replace', index=False)
    clean_conn.commit()
    clean_conn.close()

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
        processed CCS values for the entries
    """
    ccs_values = [e["ccs"] for e in entries]
    rsd = calculate_rsd(ccs_values)
    
    dt_entries = [e for e in entries if e["ccs_type"] == "DT"]
    dt_ccs_entries = [e["ccs"] for e in dt_entries]

    # Ensure that we always return a list of values with the same length as the original entries list
    # logic for handling duplicate entries (exactly two entries in group)

    if len(ccs_values) == 2:
        
        print("processing two entries")
        if rsd <= 1:
            print("two entries: RSD less than 1")
            # if RSD < 1%, simply average the values
            averaged_value = round(np.mean(ccs_values), 4)
            return [averaged_value] 
        
        # if RSD > 1%, process based on ccs_type
        else:
            print("two entries: RSD more than one continue with more")
            # Handling cases with exactly two entries and DT considerations
            if len(dt_entries) == 1:
                dt_ccs = [e["ccs"] for e in dt_entries] # If exactly one entry is DT and RSD > 1%, keep only the DT measurement
                return dt_ccs
            elif len(dt_entries) == 2:
                # If both are DT and RSD > 1%, keep both values
                return [ccs_values]  # Return the original list (already the correct length)
            else:
                # If no entries are DT and RSD > 1%, keep both values
                return [ccs_values]  # Return the original list
        
    elif len(ccs_values) > 2:
        print("processing more than two entries")
        
        if rsd <= 1:  # If RSD < 1%, average all CCS values
            print("more than two entries: RSD less than 1")
            averaged_value = round(np.mean(ccs_values), 4)
            return [averaged_value] 
    
        else:  # If RSD > 1%, attempt to remove outliers and recheck RSD
            print("more than two entries: RSD more than one continue with more")
            if len(dt_entries) >= 1:
                print("processing DT entries ONLY")
                new_dt_average = remove_outliers_and_average(dt_ccs_entries)
                return new_dt_average
            else:
                print("processing entries with no DT")
                new_value = remove_outliers_and_average(ccs_values)
                return new_value
            
    else:
        print("processing one entries directly return")
        return [ccs_values]  # Already a list


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


def clean_database(db_path):
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

    # Read data from the original database into a pandas DataFrame
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT g_id, name, adduct, mass, z, mz, ccs, smi, chem_class_label, src_tag, ccs_type, ccs_method FROM master",
        conn
    )
    conn.close()
    
    # Normalize the name to lowercase and round mz for grouping
    df['name'] = df['name'].str.lower()
    df['rounded_mz'] = df['mz'].round(0)
    df['ccs'] = df['ccs'].astype(int)
    

    # Group by name, adduct, and rounded mz
    grouped = df.groupby(['name', 'adduct', 'rounded_mz'])
    
    clean_conn = sqlite3.connect(db_path, timeout=30)
    entry_count = 0  # Initialize entry counter

    for group_key, group_df in grouped:
        
        print(f"🔴 Entries processed: {entry_count}/{len(grouped)} 🔴 ")            

        if len(group_df) > 1:
            print(f"✅ Processing group with key: {group_key} {group_df} and size: {len(group_df)}")
            
            print("converting into dictonary")
            # Convert the group DataFrame to a list of dictionaries
            group_entries = group_df.to_dict(orient='records')
            
            print("processing entries")
            # Process entries using the provided function 
            # returned either a single value of list of values
            processed_ccs = process_entries(group_entries)    
            print(f"🟠 Processed CCS: {processed_ccs} 🟠")

            for i, ccs in enumerate(processed_ccs):
                entry = {
                    'g_id' : _gen_id(group_entries[i]["name"], group_entries[i]["adduct"], ccs, group_entries[i]["ccs_type"], ""),
                    "name": group_entries[i]["name"],  # Access the i-th dictionary's "name"
                    "adduct": group_entries[i]["adduct"],
                    "mass": group_entries[i]["mass"],
                    "z": group_entries[i]["z"],
                    "mz": group_entries[i]["mz"],
                    "ccs": ccs,  # Use the i-th processed CCS value
                    "smi": group_entries[i]["smi"],
                    'chem_class_label': group_entries[i]["chem_class_label"],
                    'src_tag': group_entries[i]["src_tag"],
                    "ccs_type": group_entries[i]["ccs_type"],
                    "ccs_method": group_entries[i]["ccs_method"]
                }

                entry_df = pd.DataFrame([entry])
            
                try:
                    if os.path.exists(db_path):
                        os.remove(db_path)

                    create_clean_db(db_path)
                    clean_conn = sqlite3.connect(db_path)

                    entry_df.to_sql('master', clean_conn, if_exists='append', index=False)
                    
                    print(f"🟠 Inserted processed group with key: {group_key}")
                    entry_count += 1
                    clean_conn.commit()
                    clean_conn.close()
                except Exception as e:
                    print(f"❌ Error processing group with key: {group_key}. Error: {e}")
                    continue
        
        elif len(group_df) == 1:
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)

                create_clean_db(db_path)
                clean_conn = sqlite3.connect(db_path)
                
                # Insert the processed group into the database using pandas
                group_df.drop(columns=['rounded_mz']).to_sql('master', clean_conn, if_exists='append', index=False)
                entry_count += 1
                clean_conn.commit()
                clean_conn.close()
            except Exception as e:
                print(f"❌ Error processing group with key: {group_key}. Error: {e}")
                continue
        
        print(f"Completed processing group with key: {group_key}")
        
    print(f"Database cleaned and saved as {db_path}, entries added {entry_count}")
    print(f"✅ Clean DB done")
    

if __name__ == "__main__":
    clean_database("C3S.db", "C3S_clean.db")
