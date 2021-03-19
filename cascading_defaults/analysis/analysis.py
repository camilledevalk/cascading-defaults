from .. import plt, default_cycler
import numpy as np

def show_edgeweight_distributions(flownetworks, xaxis = 'log', bins = np.arange(-2, 13), weight='sum', subset=None, **kwargs):
    """
    A function to plot a histogram of the edgeweights of (multiple) flownetwork(s).
    """
    if type(flownetworks) != dict:
        flownetworks = {0: flownetworks}
    
    # Allows for weight = 's' or weight='sum'
    if len(weight) > 1:
        column = weight[0]
    else:
        column = weight
        
    if xaxis == 'log':
        bins=10.**bins
        bins[0] = 0
    
    # The figure
    fig, ax = plt.subplots(dpi=150)
    ax.set_prop_cycle(default_cycler)
    
    # Selecting flownetworks if there is a subset given.
    tmp_list = flownetworks.values()
    if subset:
        tmp_list = [flownetworks[i] for i in subset]
    for flownetwork in tmp_list:
        if subset and (flownetwork.label not in subset):
            continue
        flownetwork.df.hist(column=column, ax=ax, bins=bins, bottom=1, alpha=1, label=flownetwork.label, histtype='step', linewidth=1, **kwargs)
        
    if weight == 'sum':
        unit = 'EUR'
    elif weight == 'count':
        unit = 'counts'
    else:
        unit = ''
    
    plt.xlabel(f'edgeweight (in {unit})')
    plt.ylabel('$N$')
    plt.title(f'Weight of edges ({weight} of txns)')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax.set_yscale('log')
    ax.set_xscale(xaxis)
    plt.show()
    

def show_degree_distributions(flownetworks, xaxis='log', bins=np.arange(-0.5, 7, 0.5), direction='out', subset=[], absolute_relative='absolute', **kwargs):
    """
    A function to plot a histogram of the degrees of (multiple) flownetwork(s).
    """
    if type(flownetworks) != dict:
        flownetworks = {0: flownetworks}
        
    if xaxis == 'log':
        bins=10.**bins
        bins[0]=0
    
    # The figure
    fig, ax = plt.subplots(dpi=150)
    ax.set_prop_cycle(default_cycler)
    
    # Selecting flownetworks if there is a subset given.
    tmp_list = flownetworks.values()
    if subset:
        tmp_list = [flownetworks[i] for i in subset]
    for flownetwork in tmp_list:
        if subset and (flownetwork.label not in subset):
            continue
        if not hasattr(flownetwork, 'degrees_df'):
            flownetwork.calculate_degrees()
        flownetwork.degrees_df.hist(column=f'{direction} degree', ax=ax, bins=bins, bottom=1, alpha=1, label=flownetwork.label, histtype='step', linewidth=1, **kwargs)
    
    plt.xlabel(f'$k_{{{direction}}}$')
    plt.ylabel('$N$')
    plt.title(f'{direction} degree distribution')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax.set_yscale('log')
    ax.set_xscale(xaxis)
    plt.show()
    
    
def show_strength_distributions(flownetworks, xaxis = 'log', bins = np.arange(-1, 14), direction='out', subset=[], **kwargs):
    """
    A function to plot a histogram of the strengths of (multiple) flownetwork(s).
    """
    if type(flownetworks) != dict:
        flownetworks = {0: flownetworks}
        
    if xaxis == 'log':
        bins=10.**bins
        bins[0] = 0
    
    # The figure
    fig, ax = plt.subplots(dpi=150)
    ax.set_prop_cycle(default_cycler)
    
    # Selecting flownetworks if there is a subset given.
    tmp_list = flownetworks.values()
    if subset:
        tmp_list = [flownetworks[i] for i in subset]
    for flownetwork in tmp_list:
        if subset and (flownetwork.label not in subset):
            continue
        if not hasattr(flownetwork, 'flow_df'):
            flownetwork.calculate_flow()
        flownetwork.flow_df.hist(column=f'{direction}flow', ax=ax, bins=bins, bottom=1, alpha=1, label=flownetwork.label, histtype='step', linewidth=1, **kwargs)
    
    plt.xlabel('Strength of nodes (EUR)')
    plt.ylabel('$N$')
    plt.title(f'Total {direction}strength distribution')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax.set_yscale('log')
    ax.set_xscale(xaxis)
    plt.show()