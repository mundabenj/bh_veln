# import io
import os
# import pickle
# import zipfile

# import pandas as pd
# import numpy as np

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Datasets")

# Root of the local Nurses Stress dataset copy
file_path_nurses_stress = os.path.join(DATASETS_DIR, "Nurses_Stress")

# Root of the local WESAD copy; each subject has a folder S2..S17 (S1 and S12 don't exist).
file_path_wesad = os.path.join(DATASETS_DIR, "WESAD")

# PhysioNet: Wearable Device Dataset from Induced Stress and Structured Exercise Sessions
file_path_physionet = os.path.join(DATASETS_DIR, "PhysioNet_Wearable_Device")

# test if the path exists
if not os.path.exists(file_path_nurses_stress):
    raise FileNotFoundError(f"Path '{file_path_nurses_stress}' does not exist. Please check the path.")

if not os.path.exists(file_path_wesad):
    raise FileNotFoundError(f"Path '{file_path_wesad}' does not exist. Please check the path.")

if not os.path.exists(file_path_physionet):
    raise FileNotFoundError(f"Path '{file_path_physionet}' does not exist. Please check the path.")