from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Get users homebucket
current_dir = os.path.abspath(os.getcwd())

# For more beautiful plots
plt.style.use('ggplot')
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
default_cycler = cycler('color', default_colors)
mpl.rcParams['axes.prop_cycle'] = default_cycler
