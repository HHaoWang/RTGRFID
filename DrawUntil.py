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


def draw_tag_data(data: pandas.DataFrame, selected_col: Literal["PhaseAngle", "RSSI"],
                  title: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    x = data.index.tolist()
    y = data[selected_col].tolist()
    ax.plot(x, y)
    if title is not None:
        fig.suptitle(title)
    fig.show()


def draw_line(data: list):
    x = [i for i in range(len(data))]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(x, data)
    # ax.set_ylim(bottom=0, top=6.3)
    fig.show()
