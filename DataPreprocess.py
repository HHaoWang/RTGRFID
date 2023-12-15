import datetime
from typing import TypedDict

import numpy
import numpy as np
import pandas as pd
import pandas.core.groupby
from scipy.signal import savgol_filter


class TimeSpan(TypedDict):
    startTime: datetime.datetime
    endTime: datetime.datetime


def process_single_tag_data(tag_data: pandas.DataFrame, time_span: TimeSpan):
    unwrap_first = True
    padded_data: pandas.DataFrame = tag_data.loc[:, ["RSSI", "PhaseAngle"]].copy()
    if unwrap_first:
        padded_data["PhaseAngle"] = np.unwrap(padded_data["PhaseAngle"])

    epc = tag_data.iloc[0]["EPC"]
    antenna = tag_data.iloc[0]["Antenna"]

    old_idx = padded_data.index
    new_idx = pd.date_range(time_span["startTime"], time_span["endTime"], freq='50ms')
    res = (padded_data.reindex(old_idx.union(new_idx))
           .interpolate('index', limit_direction="both")
           .reindex(new_idx))
    res["EPC"] = [epc for _ in range(len(res))]
    res["Antenna"] = [antenna for _ in range(len(res))]
    resampled_data = res.copy()
    resampled_data["RSSI"] = savgol_filter(res["RSSI"], 10, 3, mode="nearest")
    if unwrap_first:
        resampled_data["PhaseAngle"] = savgol_filter(res["PhaseAngle"], 10, 3, mode="nearest")
    else:
        resampled_data["PhaseAngle"] = savgol_filter(np.unwrap(res["PhaseAngle"]), 10, 3, mode="nearest")
    return resampled_data


def process_tag_array_data(raw_data: pandas.core.groupby.DataFrameGroupBy,
                           tag_epc_array: numpy.ndarray,
                           time_span: TimeSpan) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    tag_array_data = []
    for indexRow, row in enumerate(tag_epc_array):
        tag_array_data.append([])
        for indexCol, epc in enumerate(row):
            processed_data = process_single_tag_data(raw_data.get_group(epc), time_span)
            tag_array_data[indexRow].append(processed_data)
    # DrawUntil.draw_tag_array_data(tag_array_data, selected_col="PhaseAngle")
    # DrawUntil.draw_tag_array_data(tag_array_data, selected_col="RSSI")

    rssi, phase = [], []
    length = len(tag_array_data[0][0])
    for i in range(length):
        rssi_frame, phase_frame = [], []
        for indexRow, row in enumerate(tag_array_data):
            rssi_frame_row, phase_frame_row = [], []
            for indexCol, tag_data in enumerate(row):
                rssi_frame_row.append(tag_data.iloc[i]["RSSI"])
                phase_frame_row.append(tag_data.iloc[i]["PhaseAngle"])
            rssi_frame.append(rssi_frame_row)
            phase_frame.append(phase_frame_row)
        rssi.append(rssi_frame)
        phase.append(phase_frame)
    return rssi, phase
