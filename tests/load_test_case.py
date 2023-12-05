import pandas as pd


def load_test_case(trans_flow_file_path, sheet_name="Sheet1"):
    return pd.read_excel(trans_flow_file_path, sheet_name=sheet_name)

