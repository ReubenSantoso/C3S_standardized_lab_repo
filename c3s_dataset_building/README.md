# C3S Dataset Building

This repository contains tools and utilities for building the C3S (Collision Cross-Section) database from raw JSON datasets. The C3S database consolidates collision cross-section measurements from multiple research groups into a standardized format for use in ion mobility-mass spectrometry applications.

## Overview

The C3S database building process takes raw JSON datasets containing collision cross-section measurements and creates a comprehensive SQLite database with the following features:

- **Standardized data format** from multiple research sources
- **SMILES structure lookup** via PubChem API
- **Molecular quantum numbers (MQNs)** calculation
- **Chemical classification** labeling
- **Data cleaning** and quality control
- **Multiple database versions** (raw, cleaned, and filtered)

## Folder Structure

```
c3s_dataset_building/
├── c3s_build.ipynb          # Main notebook for building the database
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── C3S.db                  # Generated database file
├── _include/               # Required files and data
│   ├── C3SDB_schema.sqlite3    # Main database schema
│   ├── mqn_schema.sqlite3      # MQN table schema
│   ├── pred_CCS_schema.sqlite3 # Predicted CCS schema
│   ├── smiles_search_cache.json # SMILES lookup cache
│   └── src_data/               # Raw JSON datasets
│       ├── zhou1016.json
│       ├── zhou0817.json
│       ├── zhen0917.json
│       └── ... (30+ datasets)
└── build_utils/            # Utility modules
    ├── __init__.py
    ├── db_init.py          # Database initialization
    ├── src_data.py         # Source data processing
    ├── smiles.py           # SMILES structure lookup
    ├── mqns.py             # MQN calculation
    ├── classification.py   # Chemical classification
    ├── clean_src.py        # Data cleaning utilities
    └── ... (other utilities)
```

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Internet connection (for PubChem API calls)

### Python Environment Setup

1. **Create a virtual environment** (recommended):
```bash
python -m venv c3s_env
source c3s_env/bin/activate  # On Windows: c3s_env\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `sqlite3` - Database operations (built-in)
- `requests` - HTTP requests for PubChem API
- `rdkit` - Chemical informatics (SMILES validation, MQN calculation)

## Quick Start

### 1. Run the Main Notebook

Open and run `c3s_build.ipynb` in Jupyter Notebook:

```bash
jupyter notebook c3s_build.ipynb
```

### 2. Execute Cells Sequentially

The notebook is organized into clear sections:

1. **Initialize DB** - Creates the database schema
2. **Populate DB** - Adds raw data from JSON files
3. **Add SMILES** - Fetches chemical structures from PubChem
4. **Add MQNs** - Calculates molecular quantum numbers
5. **Add Classification** - Applies chemical class labels
6. **Clean Database** - Creates quality-controlled versions

### 3. Expected Output

The process will generate several database files:
- `C3S.db` - Complete raw database
- `C3S_clean_smile.db` - Database with valid SMILES only
- `C3S_clean_smile_rsd.db` - Final cleaned database with RSD filtering

## Detailed Process

### Step 1: Database Initialization
Creates a new SQLite database with three main tables:
- `master` - Main data table with CCS measurements
- `mqn` - Molecular quantum numbers
- `pred_ccs` - Predicted CCS values

### Step 2: Raw Data Ingestion
Processes 30+ JSON datasets from research groups:
- Each dataset contains CCS measurements with metadata
- Data is standardized and assigned unique identifiers
- Source tags track data provenance

### Step 3: SMILES Structure Addition
- Queries PubChem API for chemical structures
- Uses cached results to minimize API calls
- Validates SMILES strings using RDKit

### Step 4: MQN Calculation
- Calculates molecular quantum numbers for each compound
- Uses RDKit for molecular descriptor calculation

### Step 5: Chemical Classification
- Applies rough chemical class labels based on compound names
- Categories include: small molecule, lipid, peptide, etc.

### Step 6: Data Cleaning
- **SMILES validation**: Removes entries with invalid structures
- **RSD filtering**: Groups compounds by name/adduct/mass and filters by relative standard deviation
- **Quality control**: Ensures data consistency and reliability

## Configuration

### Source Datasets
The `_SRC_TAGS` list in the notebook controls which datasets are included:

```python
_SRC_TAGS = [
    "zhou1016", "zhou0817", "zhen0917", "pagl0314",
    "righ0218", "nich1118", "may_0114", "moll0218",
    # ... more datasets
]
```

### Database Names
You can customize output database names:

```python
db_name = "C3S.db"                    # Main database
clean_db_name = "C3S_clean_smile.db"  # Cleaned database
```

## Data Sources

The database includes CCS measurements from 30+ research groups, including:
- Zhou et al. (2016, 2017) - Large-scale metabolite predictions
- Pagliano et al. (2014) - Small molecule measurements
- Ridgeway et al. (2018) - Lipid CCS database
- And many more...

Each dataset includes:
- Compound names and SMILES structures
- Mass-to-charge ratios (m/z)
- Collision cross-section values
- Ionization adducts
- Measurement conditions and methods

## Troubleshooting

### Common Issues

1. **RDKit installation problems**:
```bash
conda install -c conda-forge rdkit
```

2. **PubChem API rate limiting**:
- The notebook includes delays and caching to handle this
- If issues persist, increase delays in the SMILES lookup section

3. **Memory issues with large datasets**:
- Process datasets in smaller batches
- Increase system memory or use a machine with more RAM

4. **Database file permissions**:
- Ensure write permissions in the working directory
- Close any applications that might be using the database file

### Performance Tips

- **Use SSD storage** for faster database operations
- **Increase available RAM** for processing large datasets
- **Use a stable internet connection** for PubChem API calls
- **Run during off-peak hours** to avoid API rate limiting

## Output Database Schema

### Main Tables

**master table**:
- `g_id` - Unique identifier
- `name` - Compound name
- `adduct` - Ionization adduct
- `mass`, `z`, `mz`, `ccs` - Mass spectrometry data
- `smi` - SMILES structure
- `chem_class_label` - Chemical classification
- `src_tag` - Source dataset identifier
- `ccs_type` - CCS measurement type (DT, TW)
- `ccs_method` - Measurement method description

**mqn table**:
- `g_id` - Links to master table
- `r1` through `r9`, `rg10` - Ring descriptors
- `afr`, `bfr` - Aromatic/bonding features

## Contributing

To add new datasets:

1. Format your data as JSON with the required structure
2. Add the dataset file to `_include/src_data/`
3. Update the `_SRC_TAGS` list in the notebook
4. Test the build process

## License

This software is developed for research purposes. Please cite the original datasets when using this database.

## Contact

For questions or issues:
- Dylan Ross (dylan.ross@pnnl.gov) - Original development
- Reuben Santoso (reubens@uw.edu) or Amogh Bantwal (amoghb07@uw.edu) - Current maintainers

## Version History

- **v1.0** - Initial database build system
- **v1.1** - Added data cleaning utilities and RSD filtering
- **v1.2** - Enhanced SMILES validation and error handling
