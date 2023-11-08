from typing import Literal

import matplotlib.axes
import matplotlib.pyplot
import numpy
import pandas
from matplotlib import pyplot as plt


def draw_tag_array_data(tag_array_data: numpy.ndarray, selected_col: Literal["PhaseAngle", "RSSI"]) -> None:
    fig, ax = plt.subplots(ncols=tag_array_data.shape[0], nrows=tag_array_data.shape[1], figsize=(12, 9))
    fig: matplotlib.pyplot.Figure
    fig.suptitle(selected_col)

    for indexRow, row in enumerate(tag_array_data):
        for indexCol, tagData in enumerate(row):
            plot: matplotlib.axes.Axes = ax[indexRow, indexCol]
            tagData: pandas.DataFrame
            x = tagData.index.tolist()
            y = list(tagData[selected_col])
            plot.plot(x, y)
            plot.get_xaxis().set_visible(False)
    fig.show()
