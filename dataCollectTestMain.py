import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

from DrawUntil import draw_tag_data, draw_line


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


tagArray1 = np.array([
    ["E280689400005013F1509918", "E280689400005013F150B118", "E280689400005013F150A518"],
    ["E280689400004013F1509D18", "E280689400004013F150B518", "E280689400005013F150A918"],
    ["E280689400005013F1509518", "E280689400004013F150AD18", "E280689400004013F150A118"]
])
position = (0, 1)
file_path = r"C:\Users\HHao\OneDrive\学习\研究生\小论文\实验数据\实验室\11-21-2023_16h_37m_21s.csv"
data = pd.read_csv(file_path, header=2, parse_dates=["// Timestamp"], index_col="// Timestamp")
columns = []
for columnName in data.columns:
    name: str = columnName.strip()
    name = name.strip("// ")
    columns.append(name)
data.columns = columns
data = data.loc[:, ["EPC", "Antenna", "RSSI", "PhaseAngle"]].copy()
data["RSSI"] = data["RSSI"].astype(np.float32)
data["PhaseAngle"] = data["PhaseAngle"].astype(np.float32)
# data["PhaseAngle"] = savgol_filter(data["PhaseAngle"], 20, 3, mode="nearest")
# data["RSSI"] = savgol_filter(data["RSSI"], 20, 3, mode="nearest")
# data["PhaseAngle"] = np.unwrap(data["PhaseAngle"])

tag_array_raw_data = data[
    (data["EPC"].isin(tagArray1.flatten().tolist())) & (data["Antenna"] == 1)]
tag_epc = tagArray1[position[0], position[1]]
tag_raw_data = tag_array_raw_data[data["EPC"] == tag_epc]

phase = np.array(tag_raw_data["PhaseAngle"])
window_size = 10
smoothed_phase = []
for i in range(len(phase) - window_size):
    window = phase[i:i + 10]
    std = np.std(window)
    mean = np.mean(window)
    selected = np.ones(len(window), dtype=np.int8)
    max_value = np.max(window)
    min_value = np.min(window)
    for j in range(len(window)):
        if window[j] < mean - std or window[j] > mean + std or window[j] == max_value or window[j] == min_value:
            selected[j] = 0
    selected_value = window[selected]
    smoothed_phase.append(np.mean(selected_value))

smoothed_phase = np.unwrap(smoothed_phase)
draw_line(smoothed_phase)

tag_raw_data["PhaseAngle"] = np.unwrap(tag_raw_data["PhaseAngle"])
draw_tag_data(tag_raw_data, "PhaseAngle", str(tag_epc))
