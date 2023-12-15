import pathlib

import pandas
import pandas as pd
import numpy as np

from DataPreprocess import TimeSpan, process_tag_array_data
from utils import get_one_hot


def get_data_in_pandas(data_directory: str) -> dict:
    base_path = pathlib.Path(data_directory)
    classes_directory_list = [x for x in base_path.iterdir() if x.is_dir()]
    dataset = {}
    for classes_directory in classes_directory_list:
        dataset[classes_directory.name] = []
        for file in classes_directory.iterdir():
            if not file.is_file():
                continue
            data = pd.read_csv(file, header=2, parse_dates=["// Timestamp"], index_col="// Timestamp")
            columns = []
            for columnName in data.columns:
                name: str = columnName.strip()
                name = name.strip("// ")
                columns.append(name)
            data.columns = columns
            data: pd.DataFrame = data.loc[:, ["EPC", "Antenna", "RSSI", "PhaseAngle"]].copy()
            data["RSSI"] = data["RSSI"].astype(np.float32)
            data["PhaseAngle"] = data["PhaseAngle"].astype(np.float32)
            data = data.drop_duplicates()
            dataset[classes_directory.name].append(data)
    return dataset


def preprocess_data_in_pandas(raw_data: pandas.DataFrame, tag_epc_array1, tag_epc_array2, seq_len: int):
    tag_array1_raw_data = raw_data[
        (raw_data["EPC"].isin(tag_epc_array1.flatten().tolist())) & (raw_data["Antenna"] == 1)]
    tag_array1_raw_data.sort_index(inplace=True)
    min_start_time = tag_array1_raw_data.index[0]
    max_end_time = tag_array1_raw_data.index[-1]

    tag_array2_raw_data = raw_data[
        (raw_data["EPC"].isin(tag_epc_array2.flatten().tolist())) & (raw_data["Antenna"] == 2)]
    tag_array2_raw_data.sort_index(inplace=True)

    min_start_time = tag_array2_raw_data.index[0] if tag_array2_raw_data.index[0] < min_start_time else min_start_time
    max_end_time = tag_array2_raw_data.index[-1] if tag_array2_raw_data.index[-1] > max_end_time else max_end_time

    last_span = TimeSpan(startTime=min_start_time, endTime=max_end_time)

    group1 = tag_array1_raw_data.groupby("EPC")
    group2 = tag_array2_raw_data.groupby("EPC")
    rssi1, phase1 = process_tag_array_data(group1, tag_epc_array1, last_span)
    rssi2, phase2 = process_tag_array_data(group2, tag_epc_array2, last_span)

    processed_data = []
    for i in range(len(rssi1)):
        processed_data.append(np.array([
            [rssi1[i], phase1[i]],
            [rssi2[i], phase2[i]]
        ]))

    if len(processed_data) > seq_len:
        processed_data = processed_data[:seq_len]
    if len(processed_data) < seq_len:
        last = processed_data[-1]
        for i in range(seq_len - len(processed_data)):
            processed_data.append(last.copy())

    return np.array(processed_data)


def get_data(data_directory: str, tag_epc_array1, tag_epc_array2, seq_len: int):
    dataset_in_pandas = get_data_in_pandas(data_directory)
    processed_data = []
    labels = []
    keys_in_one_hot = get_one_hot(dataset_in_pandas.keys())
    for key in dataset_in_pandas:
        for data in dataset_in_pandas[key]:
            processed_data.append(preprocess_data_in_pandas(data, tag_epc_array1, tag_epc_array2, seq_len))
            labels.append(keys_in_one_hot[key])
    return np.array(processed_data), np.array(labels)
