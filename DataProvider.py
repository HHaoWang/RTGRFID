import datetime
import time
from queue import Queue
from threading import Thread
from typing import TypedDict

import numpy
import numpy as np
import pandas as pd
import pandas.core.groupby
from scipy.signal import savgol_filter

# noinspection PyUnresolvedReferences
import DrawUntil
from DataCollector import DataCollector


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
    tag_array_data = np.asarray(
        [raw_data.get_group(tag_epc_array.flatten().tolist()[i]) for i in range(tag_epc_array.size)],
        dtype=object).reshape(tag_epc_array.shape)

    for indexRow, row in enumerate(tag_array_data):
        for indexCol, tag_data in enumerate(row):
            tag_array_data[indexRow][indexCol] = process_single_tag_data(tag_data, time_span)

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


class DataProvider:
    unprocessed_data_packages: Queue[pandas.DataFrame] = Queue()
    processed_data: Queue[numpy.ndarray] = Queue()
    """
    处理后的数据队列，用于作为滑动窗口的数据源\n
    队列中每个元素的shape = (2,2,tag_epc_array.shape)\n
    表示有两个阵列，每个阵列有RSSI和Phase种数据，每种数据有tag_epc_array形状的数据阵列
    """

    data_window_size: int = 10
    tag_epc_array1: numpy.ndarray | None = None
    tag_epc_array2: numpy.ndarray | None = None
    __has_more_data_package: bool = True
    __has_more_data: bool = True
    __current_window: list = []

    def receive_data(self, data: pandas.DataFrame, has_more_data: bool) -> None:
        self.unprocessed_data_packages.put(data)
        self.__has_more_data_package = has_more_data

    def register_collector(self, collector: DataCollector):
        collector.register_receive(self.receive_data)

    def process_data_package(self):
        if self.unprocessed_data_packages.empty() or self.tag_epc_array1 is None or self.tag_epc_array2 is None:
            return
        raw_data = self.unprocessed_data_packages.get()

        tag_array1_raw_data = raw_data[
            (raw_data["EPC"].isin(self.tag_epc_array1.flatten().tolist())) & (raw_data["Antenna"] == 1)]
        tag_array1_raw_data.sort_index(inplace=True)
        min_start_time = tag_array1_raw_data.index[0]
        max_end_time = tag_array1_raw_data.index[-1]

        tag_array2_raw_data = raw_data[
            (raw_data["EPC"].isin(self.tag_epc_array2.flatten().tolist())) & (raw_data["Antenna"] == 2)]
        tag_array1_raw_data.sort_index(inplace=True)
        min_start_time = tag_array2_raw_data.index[0] if tag_array2_raw_data.index[0] < min_start_time else min_start_time
        max_end_time = tag_array2_raw_data.index[-1] if tag_array2_raw_data.index[-1] > max_end_time else max_end_time

        last_span = TimeSpan(startTime=min_start_time, endTime=max_end_time)

        group1 = tag_array1_raw_data.groupby("EPC")
        group2 = tag_array2_raw_data.groupby("EPC")
        rssi1, phase1 = process_tag_array_data(group1, self.tag_epc_array1, last_span)
        rssi2, phase2 = process_tag_array_data(group2, self.tag_epc_array2, last_span)

        for i in range(len(rssi1)):
            self.processed_data.put(np.array([
                [rssi1[i], phase1[i]],
                [rssi2[i], phase2[i]]
            ]))

    def start_func(self):
        while self.__has_more_data_package or not self.unprocessed_data_packages.empty():
            if self.unprocessed_data_packages.empty():
                time.sleep(0.1)
                continue
            while not self.unprocessed_data_packages.empty():
                self.process_data_package()
        self.__has_more_data = False

    def start(self):
        thread = Thread(target=self.start_func)
        thread.start()

    def get_data_frame(self):
        while self.__has_more_data or not self.processed_data.empty():
            if self.processed_data.empty():
                continue
            yield self.processed_data.get()

    def get_data_window(self):
        if self.data_window_size <= 0:
            raise Exception("窗口大小必须为正整数")
        while self.__has_more_data or self.processed_data.qsize() > 0:
            if self.processed_data.qsize() == 0:
                time.sleep(0.1)
                continue
            if len(self.__current_window) == self.data_window_size:
                self.__current_window.pop(0)
                self.__current_window.append(self.processed_data.get())
                yield self.__current_window
            else:
                while len(self.__current_window) < self.data_window_size:
                    self.__current_window.append(self.processed_data.get())
                yield self.__current_window
