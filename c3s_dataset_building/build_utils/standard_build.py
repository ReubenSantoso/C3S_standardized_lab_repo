"""
    c3sdb/build_utils/standard_build.py

    Dylan Ross (dylan.ross@pnnl.gov)
    
    standard build script for the CCSbase database (C3S.db)
    
    - use command: `python3 -m c3sdb.build_utils.standard_build`
    - creates files:
        - `C3S.db`: database
        - `smiles_search_cache.json`: cached values for searching for SMILES structures
        - `C3S_clean.db`: cleaned database 
"""

# TODO
# need to calculate mqns for C3S_clean
# add in new dataset
# re-train model

import sqlite3
import os

import requests

from build_utils.db_init import create_db
from build_utils.src_data import add_dataset
from build_utils.smiles import (
    load_smiles_search_cache,
    save_smiles_search_cache,
    add_smiles_to_db,
)
from build_utils.mqns import add_mqns_to_db
from build_utils.classification import label_class_byname
from build_utils.clean_src import clean_database


# source datasets to include
_SRC_TAGS = [
    "zhou1016",
    "zhou0817",
    "zhen0917",
    "pagl0314",
    "righ0218",
    "nich1118",
    "may_0114",
    "moll0218",
    "hine1217",
    "hine0217",
    "hine0817",
    "groe0815",
    "bijl0517",
    "stow0817",
    "hine0119",
    "leap0219",
    "blaz0818",
    # "vasi0120",
    "tsug0220",
    "lian0118",
    "teja0918",
    "pola0620",
    "dodd0220",
    "celm1120",
    "belo0321",
    "ross0422",
    "baker0524", #new
    "mull_1223", #new
    "palm_0424", #new
]


def _main():
    # database file
    dbf = "C3S.db"
    # create the database
    print("initializing database ...", end=" ")
    create_db(dbf)
    print("done")

    # connect to database
    con = sqlite3.connect(dbf)
    cur = con.cursor()

    # add source datasets
    print("adding source datasets ...")
    n_entries = 0
    for src_tag in _SRC_TAGS:
        n_added = add_dataset(cur, src_tag)
        n_entries += n_added
        print(f"\tsrc_tag: {src_tag} n_added: {n_added}")
    print(f"\ttotal entries: {n_entries}")
    print("... done")

    # add SMILES structures
    print("adding SMILES structures ...")
    smiles_cache_file = "smiles_search_cache.json"
    # if a local copy of the SMILES search cache does not exist, grab the
    # built-in copy from the package
    if not os.path.isfile(smiles_cache_file):
        smiles_search_cache = load_smiles_search_cache(cache_file_name=None)
    else:
        # load the local copy if it exists
        smiles_search_cache = load_smiles_search_cache(
            cache_file_name=smiles_cache_file 
        )
    sess = requests.Session()
    n_smiles, n_requests = add_smiles_to_db(cur, sess, smiles_search_cache)
    print(f"\tSMILES structures added: {n_smiles}")
    print(f"\tweb requests sent: {n_requests}")
    print("... done")
    # save the search cache
    save_smiles_search_cache(smiles_search_cache, smiles_cache_file)
    # add MQNs
    print("adding MQNs to database entries ...")
    n_mqns = add_mqns_to_db(cur)
    print(f"\tentries with MQNs: {n_mqns}")
    print("... done")
    # add rough chemical classification labels
    print("adding rough chemical classification labels to database entries ...")
    label_class_byname(cur)
    print("... done")
    # commit changes to database and close
    con.commit()
    con.close()
    
    # clean database
    clean_database("C3S.db", "C3S_clean.db")
    # connect to clean database
    clean_con = sqlite3.connect("C3S_clean.db")
    clean_cur = clean_con.cursor()
    # add MQNs to clean databse
    print("adding MQNs to cleaned database entries ...")
    n_mqns_clean = add_mqns_to_db(clean_cur)
    print(f"\tentries with MQNs in clean database: {n_mqns_clean}")
    # commit changes to clean database and close
    clean_con.commit()
    clean_con.close()

if __name__ == "__main__":
    _main()
