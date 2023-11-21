import numpy as np
from FileDataCollector import FileDataCollector
from DataProvider import DataProvider
from RTGRFID import RTGRFID

dataPath = r"C:\Users\HHao\OneDrive\学习\研究生\小论文\实验数据\2023-11-04_13-20-41.csv"
numberOfRows = 3
numberOfCols = 3
tagArray1 = np.array([
    ["E280689400005013F1509918", "E280689400005013F150B118", "E280689400005013F150A518"],
    ["E280689400004013F1509D18", "E280689400004013F150B518", "E280689400005013F150A918"],
    ["E280689400005013F1509518", "E280689400004013F150AD18", "E280689400004013F150A118"]
])

tagArray2 = np.array([
    ["E280689400005013F150D118", "E280689400004013F151C517", "E280689400005013F150C518"],
    ["E280689400004013F150CD18", "E280689400004013F150B918", "E280689400004013F150C118"],
    ["E280689400004013F150D518", "E280689400005013F150BD18", "E280689400005013F150C918"]
])

dataProvider = DataProvider()
fileDataCollector = FileDataCollector()
fileDataCollector.file_path = dataPath
dataProvider.register_collector(fileDataCollector)
dataProvider.tag_epc_array1 = tagArray1
dataProvider.tag_epc_array2 = tagArray2
dataProvider.start()
fileDataCollector.start_collect()

values = dataProvider.get_data_window()
value = next(values)
print(value.shape)

model = RTGRFID()
model.forward(value)
