from typing import Callable

import numpy as np
import pandas
import pandas as pd

from DataCollector import DataCollector


class FileDataCollector(DataCollector):
    on_collected_data: Callable[[pandas.DataFrame, bool], None] | None = None
    file_path: str | None = None

    def register_receive(self, on_collected_data: Callable[[pandas.DataFrame, bool], None]):
        self.on_collected_data = on_collected_data

    def start_collect(self) -> None:
        if self.file_path is None or not callable(self.on_collected_data):
            return
        data = pd.read_csv(self.file_path, header=2, parse_dates=["// Timestamp"], index_col="// Timestamp")
        columns = []
        for columnName in data.columns:
            name: str = columnName.strip()
            name = name.strip("// ")
            columns.append(name)
        data.columns = columns
        data = data.loc[:, ["EPC", "Antenna", "RSSI", "PhaseAngle"]].copy()
        data["RSSI"] = data["RSSI"].astype(np.float32)
        data["PhaseAngle"] = data["PhaseAngle"].astype(np.float32)
        self.on_collected_data(data, False)
