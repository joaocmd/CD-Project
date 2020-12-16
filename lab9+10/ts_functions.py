import itertools
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import sklearn.metrics as metrics
import config as cfg
import datetime as dt
import matplotlib.colors as colors


NR_COLUMNS: int = 3
HEIGHT: int = 4


def plot_series(series, ax: plt.Axes = None, title: str = '', x_label: str = '', y_label: str = '',
                percentage=False, show_std=False):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    if isinstance(series, dict):
        legend: list = []
        i = 0
        for name in series.keys():
            y = series[name]
            ax.set_xlim(y.index[0], y.index[-1])
            std = y.std()
            ax.plot(y, c=cfg.ACTIVE_COLORS[i], label=name)
            if show_std:
                y1 = y.add(-std)
                y2 = y.add(std)
                ax.fill_between(y.index, y1.values, y2.values, color=cfg.ACTIVE_COLORS[i], alpha=0.2)
            i += 1
            legend.append(name)
        ax.legend(legend)
    else:
        ax.plot(series)

    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
