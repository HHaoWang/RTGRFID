import numpy
import pandas as pd
import pandas.core.groupby
import datetime
from typing import Tuple


def read_parse_raw_data(file_path: str, tag_epc_array: numpy.ndarray[str], antenna: int) \
        -> Tuple[pandas.core.groupby.DataFrameGroupBy, datetime.datetime, datetime.datetime]:
    data = pd.read_csv(file_path, header=2, parse_dates=["// Timestamp"], index_col="// Timestamp")
    columns = []
    for columnName in data.columns:
        name: str = columnName.strip()
        name = name.strip("// ")
        columns.append(name)
    data.columns = columns
    data = data.drop(["TID", "Frequency", "Hostname", "DopplerFrequency"], axis=1)

    tag_array_raw_data = data[(data["EPC"].isin(tag_epc_array.flatten().tolist())) & (data["Antenna"] == antenna)]
    tag_array_raw_data.sort_index(inplace=True)
    if len(tag_array_raw_data) <= 0:
        raise Exception(f"输入的文件没有数据：{file_path}")
    min_start_time = tag_array_raw_data.index[0]
    max_end_time = tag_array_raw_data.index[-1]
    group = tag_array_raw_data.groupby("EPC")
    return group, min_start_time, max_end_time
