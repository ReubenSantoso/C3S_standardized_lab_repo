import unittest
from clean_src import calculate_rsd, remove_outliers_and_average, process_entries, create_clean_db, clean_database
import os
import sqlite3
import numpy as np
import pandas as pd 

# python3 -m unittest -v test_clean_src.py
# -m indicator used to let python know to run this file as a script instead of a module
# -v to break down the test process

# python3 -m unittest -v test_clean_src.TestCleanSrc.test_process_entries
# to run a specific test


TEST_FILES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unit_test_files")

def expected_remove_outliers(values):
    mean, std = np.mean(values), np.std(values)
    print(f"Mean of Values: {mean}. STD of Values: {std}")
    
    filtered_values = [value for value in values if abs(value - mean) <= std]
    
    while calculate_rsd(filtered_values) > 1:
        mean, std = np.mean(filtered_values), np.std(filtered_values)
        filtered_values = [value for value in filtered_values if abs(value - mean) <= std]
    
    return filtered_values

def df_to_csv(entries, expected_result, file_name):
    unique_ccs = set(expected_result) # create a set of unique CCS values to avoid duplicates
        
    # Initialize an empty DataFrame to store the cleaned entries
    clean_entry = pd.DataFrame()

    # Loop through the entries and expected_result
    for entry, ccs_value in zip(entries, expected_result):
        if ccs_value in unique_ccs:
            # Add the CCS value to the entry
            entry["ccs"] = ccs_value
            
            # Convert the entry to a DataFrame and append it to clean_entry
            entry_df = pd.DataFrame([entry])
            clean_entry = pd.concat([clean_entry, entry_df], ignore_index=True)
            
            # Remove the CCS value from the unique_ccs set to avoid duplicates
            unique_ccs.remove(ccs_value)

    # Save the cleaned DataFrame to a CSV file
    output_csv_path = os.path.join(TEST_FILES_PATH, file_name)
    clean_entry.to_csv(output_csv_path, index=False)
    
    print("Cleaned DataFrame:")
    print(clean_entry)


class TestC3SDatabase(unittest.TestCase):

    #/Users/reubensantoso/Xu_Lab_Files/c3sdb/c3sdb/build_utils/test_clean_src.py
    def setUp(self):
        INCLUDE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        # Set up any initialization code here
        self.db_path = os.path.join(INCLUDE_PATH ,"C3S.db")
        self.conn = sqlite3.connect(self.db_path)

    def tearDown(self):
        # Clean up resources, if needed
        self.conn.close()

    def test_database_connection(self):
        # Test if the database connection is established
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM master LIMIT 1")
        row = cursor.fetchone()
        self.assertIsNotNone(row, "Expected at least one row in the 'master' table")

    def test_row_existence(self):
        # Test if rows exist in the 'master' table
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM master")
        count = cursor.fetchone()[0]
        self.assertGreater(count, 0, "Expected more than 0 rows in the 'master' table")

    def test_create_clean_db(self):
        create_clean_db(self.clean_db_path)
        self.assertTrue(os.path.exists(self.clean_db_path))
    
    def test_clean_database(self):
        clean_database(self.db_path, self.clean_db_path)
        self.assertTrue(os.path.exists(self.clean_db_path))
        conn = sqlite3.connect(self.clean_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM master")
        rows = cursor.fetchall()
        conn.close()
        self.assertTrue(len(rows) > 0)

class TestCleanSrc(unittest.TestCase):

    def test_calculate_rsd(self):
        values = [10, 12, 23, 23, 16, 23, 21, 16]
        result = calculate_rsd(values)
        expected_result = np.std(values) / np.mean(values) * 100
        self.assertAlmostEqual(result, expected_result)

    def test_remove_outliers_and_average(self):
        values = [10, 12, 23, 23, 16, 23, 21, 16]
        actual_result = remove_outliers_and_average(values)
        filtered_values = expected_remove_outliers(values)

        expected_result = round(np.mean(filtered_values), 4)
        print(f"The filtered values: {filtered_values}")
        print(f"Expected result: {expected_result}. Actual result: {actual_result}")

        self.assertEqual(actual_result, expected_result)

    def test_remove_outliers_random(self):
        values = np.random.randint(1, 100, 10000) + 50  # Ensure we get large values around 50
        actual_result = remove_outliers_and_average(values)
        filtered_values = expected_remove_outliers(values)
        
        expected_result = round(np.mean(filtered_values), 4)
        print(f"The filtered values: {filtered_values}")
        print(f"Expected result: {expected_result}. Actual result: {actual_result}")

        self.assertEqual(actual_result, expected_result)

    def test_process_entries_with_dt(self):
        # Testing for more than two entries with all DT

        testFile = os.path.join(TEST_FILES_PATH, "tryptophan_duplicates.csv")
        df = pd.read_csv(testFile)        
        entries = df.to_dict(orient='records')
        actual_result = process_entries(entries) 
        
        ccs_values = [e["ccs"] for e in entries]
        rsd = calculate_rsd(ccs_values)

        dt_entries = [e for e in entries if e["ccs_type"] == "DT"]
        dt_ccs_entries = [e["ccs"] for e in dt_entries]
        
        # Testing for more than two entries with all DT
        if rsd <= 1:
            expected_result = [round(np.mean(ccs_values), 4)] * len(entries)
        else:
            if len(dt_entries) >= 1:
                new_dt_entries = remove_outliers_and_average(dt_ccs_entries)
                expected_result = [new_dt_entries] * len(entries)
            else:
                new_value = remove_outliers_and_average(ccs_values)
                expected_result = [new_value] * len(entries)

        df_to_csv(entries, expected_result, "cleaned_tryptophan_duplicates.csv")
        
        self.assertEqual(actual_result, expected_result)
    
    def test_process_entries_with_dt_2(self):
        # Testing for more than two entries with all DT

        testFile = os.path.join(TEST_FILES_PATH, "TG56_duplicates_new.csv")
        df = pd.read_csv(testFile)
        entries = df.to_dict(orient='records')
        actual_result = process_entries(entries) 
        
        ccs_values = [e["ccs"] for e in entries]
        rsd = calculate_rsd(ccs_values)

        dt_entries = [e for e in entries if e["ccs_type"] == "DT"]
        dt_ccs_entries = [e["ccs"] for e in dt_entries]
        
        # Testing for more than two entries with all DT
        if rsd <= 1:
            expected_result = [round(np.mean(ccs_values), 4)] * len(entries)
        else:
            if len(dt_entries) >= 1:
                new_dt_entries = remove_outliers_and_average(dt_ccs_entries)
                expected_result = [new_dt_entries] * len(entries)
            else:
                new_value = remove_outliers_and_average(ccs_values)
                expected_result = [new_value] * len(entries)

        df_to_csv(entries, expected_result, "cleaned_TG56_duplicates.csv")
        
        self.assertEqual(actual_result, expected_result)

    def test_process_entries_mixed_2(self):
        # Testing for more than two entries with mixed calibrations

        testFile = os.path.join(TEST_FILES_PATH, "PG16_duplicates.csv")
        df = pd.read_csv(testFile)
        entries = df.to_dict(orient='records')
        actual_result = process_entries(entries) 
        
        ccs_values = [e["ccs"] for e in entries]
        rsd = calculate_rsd(ccs_values)

        dt_entries = [e for e in entries if e["ccs_type"] == "DT"]
        dt_ccs_entries = [e["ccs"] for e in dt_entries]
        
        # Testing for more than two entries with all DT
        if rsd <= 1:
            expected_result = [round(np.mean(ccs_values), 4)] * len(entries)
        else:
            if len(dt_entries) >= 1:
                new_dt_entries = remove_outliers_and_average(dt_ccs_entries)
                expected_result = [new_dt_entries] * len(entries)
            else:
                new_value = remove_outliers_and_average(ccs_values)
                expected_result = [new_value] * len(entries)

        df_to_csv(entries, expected_result, "cleaned_PG16_duplicates.csv")
        
        self.assertEqual(actual_result, expected_result)

    def test_process_entries_nonDT(self):
        # Testing for two entries with no DT

        testFile = os.path.join(TEST_FILES_PATH, "13podocarpatrien.csv")
        df = pd.read_csv(testFile)
        entries = df.to_dict(orient='records')
        actual_result = process_entries(entries) 
        
        ccs_values = [e["ccs"] for e in entries]
        rsd = calculate_rsd(ccs_values)
        
        if rsd <= 1:
            # if RSD < 1%, simply average the values
            expected_result = [round(np.mean(ccs_values), 4)] * len(entries)
        
        # if RSD > 1%, process based on ccs_type
        else:
            expected_result = ccs_values

        df_to_csv(entries, expected_result, "cleaned_13podocarpatrien.csv")
        
        self.assertEqual(actual_result, expected_result)
    
    def test_process_entries_nonDT_2(self):
        # Testing for two entries with no DT

        testFile = os.path.join(TEST_FILES_PATH, "trytophan_2entries.csv")
        df = pd.read_csv(testFile)
        entries = df.to_dict(orient='records')
        actual_result = process_entries(entries) 
        
        ccs_values = [e["ccs"] for e in entries]
        rsd = calculate_rsd(ccs_values)
        
        if rsd <= 1:
            # if RSD < 1%, simply average the values
            expected_result = [round(np.mean(ccs_values), 4)] * len(entries)
        # if RSD > 1%, process based on ccs_type
        else:
            expected_result = ccs_values

        df_to_csv(entries, expected_result, "cleaned_trytophan_2entries.csv")
        
        self.assertEqual(actual_result, expected_result)

if __name__ == '__main__':
    unittest.main()
