import re

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .. import plt, default_cycler


def flow_over_time(simulations, xlim=None, absolute_relative='relative', linestyle_exo=None):
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12.5,5), dpi=150)
    ax1.set_prop_cycle(default_cycler)
    
    for simulation in simulations.values():
        if absolute_relative == 'absolute':
            size_p = simulation.size_p
        elif absolute_relative == 'relative':
            size_p = simulation.size_p_relative
        linestyle = linestyle_exo if (linestyle_exo and simulation.strategy.has_exogenous) else '-'
        ax1.plot(size_p, label=f'{simulation.label}', linewidth=3, linestyle=linestyle)
    
    # Draw line of total obliagtions (only needed when plotting absolute)
    if absolute_relative == 'absolute':
        ylabel = 'Resulting flow (EUR)'
        total_flow = simulation.total_flow
        ax1.plot(simulation.total_flow)
    else:
        ylabel = 'Resulting flow (compared to available)'
        
    plt.legend()
    
    ax1.set_xlim(0,None)
    
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel('Iteration')

    fig.tight_layout()
    title = f'Resulting flow of simulations'
    plt.title(title)
    plt.show()
    
    
def show_results(simulations):
    data = [
        (
            np.mean(simulation.size_p[-int(len(simulation.size_p)/10):]), # The mean of the last 90% of the run
            np.mean(simulation.size_p[-int(len(simulation.size_p)/10):]/simulation.total_flow),
            np.std(simulation.size_p[-int(len(simulation.size_p)/10):]),
            np.std(simulation.size_p[-int(len(simulation.size_p)/10):]/simulation.total_flow),
            len(simulation.all_defaults[0]),
            simulation.all_defaults_df.reset_index('stage').mean().values[0],
            simulation.all_defaults_df.reset_index('stage').std().values[0],
            simulation.strategy.label,
            simulation.strategy.has_exogenous
        ) for simulation in simulations.values()
    ]
    
    df = (
        pd.DataFrame(
            data,
            columns=[
                'Eventual size p (abs)',
                'Eventual size p (rel)',
                'Std eventual size (abs)',
                'Std eventual size (rel)',
                'Count of not-defaulted',
                'Mean stage of default',
                'Std stage of default',
                'Strategy',
                'Exogenous'
            ]
        )
        .set_index(['Strategy'])
        .sort_index()
        .assign(color=lambda df: np.where(
            [bool(len(re.findall('XR', i))) for i in df.index.to_list()],
            list(default_cycler)[0]['color'],
            list(default_cycler)[1]['color']))
    )

    legend_elements = [mpatches.Patch(facecolor=list(default_cycler)[0]['color'], label='XR'),
                       mpatches.Patch(facecolor=list(default_cycler)[1]['color'], label='R')]
    
    fig, ax = plt.subplots(nrows=2, figsize=(14,7), dpi=150, sharex=True)
    
    simulation = list(simulations.values())[0]
    plt.suptitle(f'Results of Simulations (run: {simulation.label_of_run})')

    (
        (100*df)
        .loc[:,'Eventual size p (rel)']
        .plot.bar(
            rot=45,
            ax=ax[0],
            yerr=df['Std eventual size (rel)'],
            capsize=5,
            color=df['color']
        )
    )

    ax[0].set_ylabel('Eventual size economy (%)')

    ax[0].legend(handles=legend_elements)

    (
        df
        .loc[:,'Count of not-defaulted']
        .plot.bar(
            rot=45,
            ax=ax[1],
            yerr=df['Count of not-defaulted']/np.sqrt(df['Count of not-defaulted']),
            capsize=5,
            color=df['color']
        )
    )

    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right')
    ax[1].set_ylabel('Count of not-defaulted')


    plt.tight_layout()
    plt.show()
    
    
    # Incoming flows figure
    plt.figure(figsize=(14,7), dpi=150)
    colors = {}
    for simulation in simulations.values():
        nonzero = np.array(simulation.L.sum(axis=0)).flatten().nonzero()[0]
        in_strengths_L = np.array(simulation.L.sum(axis=0)).flatten()[nonzero]
        in_strengths_p = np.array(simulation.p.sum(axis=0)).flatten()[nonzero]

        ratios = 100*(in_strengths_p / in_strengths_L)
        nonzeros = ratios.nonzero()[0].shape

        if simulation.strategy.strategy in colors:
            color = colors[simulation.strategy.label]

        label = simulation.strategy.label

        mean = plt.scatter(nonzeros, np.mean(ratios[ratios>0]), label=label, color=None, marker='x')
        color = mean.get_facecolor()[0]
        props = dict(color=color, linestyle='-')
        box = plt.boxplot(
            ratios[ratios>0],
            positions=nonzeros,
            widths=1e4,
            manage_ticks=False,
            showfliers=False,
            boxprops=props,
            capprops=props,
            whiskerprops=props,
            medianprops=props
        )    

    plt.xlabel('Number of nonzero instrengths after simulation')
    plt.ylabel('Average size of company after simluation (% incoming)')
    plt.title(f'Results of different strategies (incoming flows), run: {simulation.label_of_run}')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
def show_reserves_distribution_at_maturity(simulations):
    plt.figure(figsize=(8,6), dpi=150)

    for simulation in simulations.values():
        plt.hist(simulation.reserves, bins=10**np.arange(0,13), label=simulation.label, density=True, histtype='step', linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Reserves at maturity (EUR)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
def show_reserves_at_maturity(simulations):
    data = []
    index = []
    for simulation in simulations.values():
        reserves = simulation.reserves

        defaulted_nodes = simulation.nodes_currently_in_default
        not_defaulted_nodes = np.setdiff1d(np.arange(0,reserves.shape[0]), defaulted_nodes)

        reserves_defaulted_nodes = reserves[defaulted_nodes].sum()
        reserves_not_defaulted_nodes = reserves[not_defaulted_nodes].sum()

        data.append(
            [reserves_not_defaulted_nodes.sum()/1e9, reserves_defaulted_nodes.sum()/1e9]
        )
        index.append(simulation.label)

    df = pd.DataFrame(
        data,
        index=index,
        columns=['Healthy nodes','Defaulted nodes']
    )

    fig, ax = plt.subplots(figsize=(8,5), dpi=150)
    df.plot.bar(ax=ax)
    plt.title('Reserves at end of simluation')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_ylabel('Reserves in B EUR')
    plt.show()
    
    
def show_degrees(simulations, degrees_bins=10**np.arange(0,6,0.5), density=True):
    simulation = list(simulations.values())[0]
    L_unweighted = simulation.L.copy()
    nonzeros = L_unweighted.data.nonzero()[0]
    L_unweighted.data[nonzeros] = np.ones(nonzeros.shape)

    in_degrees_L = np.array(L_unweighted.sum(axis=0)).flatten()
    out_degrees_L = np.array(L_unweighted.sum(axis=1)).flatten()
    
    fig, ax = plt.subplots(figsize=(12,6), ncols=2, nrows=1, dpi=150)
    
    ax[0].hist(in_degrees_L, bins=degrees_bins, density=density, label='Obligations-matrix', alpha=0.5)
    ax[0].set_xlabel('In degree')
    ax[0].set_ylabel('Density')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_title('In degrees')

    ax[1].hist(out_degrees_L, bins=degrees_bins, density=density, label='Obligations-matrix', alpha=0.5)
    ax[1].set_xlabel('Out degree')
    ax[1].set_ylabel('Density')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title('Out degrees')
    
    for simulation in simulations.values():
        p_unweighted = simulation.p.copy()
        nonzeros = p_unweighted.data.nonzero()[0]
        p_unweighted.data[nonzeros] = np.ones(nonzeros.shape)

        in_degrees = np.array(p_unweighted.sum(axis=0)).flatten()
        ax[0].hist(in_degrees, bins=degrees_bins, histtype='step', density=density, label=f'{simulation.label}', linewidth=2)

        out_degrees = np.array(p_unweighted.sum(axis=1)).flatten()
        ax[1].hist(out_degrees, bins=degrees_bins, histtype='step', density=density, label=f'{simulation.label}', linewidth=2) 
    
    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    
def show_strengths(simulations, strengths_bins=10**np.arange(0,12), density=True):
    simulation = list(simulations.values())[0]
    
    nonzeros_in = np.array(simulation.L.sum(axis=0)).flatten().nonzero()[0]
    in_strengths_L = np.array(simulation.L.sum(axis=0)).flatten()[nonzeros_in]

    nonzeros_out = np.array(simulation.L.sum(axis=1)).flatten().nonzero()[0]
    out_strengths_L = np.array(simulation.L.sum(axis=1)).flatten()[nonzeros_out]
    
    fig, ax = plt.subplots(figsize=(12,6), ncols=2, nrows=1, dpi=150)
    
    ax[0].hist(in_strengths_L, bins=strengths_bins, density=density, label='Obligations-matrix', alpha=0.5)
    ax[0].set_xlabel('In strength (EUR)')
    ax[0].set_ylabel('Density')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_title('In strengths (EUR)')

    ax[1].hist(out_strengths_L, bins=strengths_bins, density=density, label='Obligations-matrix', alpha=0.5)
    ax[1].set_xlabel('Out strength (EUR)')
    ax[1].set_ylabel('Density')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title('Out strengths (EUR)')
    
    for simulation in simulations.values():
        p_unweighted = simulation.p.copy()
        nonzeros = p_unweighted.data.nonzero()[0]
        p_unweighted.data[nonzeros] = np.ones(nonzeros.shape)

        in_strengths = np.array(simulation.p.sum(axis=0)).flatten()[nonzeros_in]
        ax[0].hist(in_strengths, bins=strengths_bins, histtype='step', density=density, label=f'{simulation.label}', linewidth=2)

        out_strengths = np.array(simulation.p.sum(axis=1)).flatten()[nonzeros_out]
        ax[1].hist(out_strengths, bins=strengths_bins, histtype='step', density=density, label=f'{simulation.label}', linewidth=2)
    
    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    
def show_ratios_payments(simulations, density=True):
    simulation = list(simulations.values())[0]
    L = simulation.L

    def calculate_nonzeros_and_strengths(L):
        # Nonzero entries
        nonzeros_in = np.array(L.sum(axis=0)).flatten().nonzero()[0]
        nonzeros_out = np.array(L.sum(axis=1)).flatten().nonzero()[0]

        # Correct for the sinknode
        nonzeros_in = nonzeros_in[np.argwhere(nonzeros_in!=0)]
        nonzeros_out = nonzeros_out[np.argwhere(nonzeros_out!=0)]

        # Strengths
        in_strengths_L = np.array(L.sum(axis=0)).flatten()[nonzeros_in]
        out_strengths_L = np.array(L.sum(axis=1)).flatten()[nonzeros_out]

        return nonzeros_in, nonzeros_out, in_strengths_L, out_strengths_L

    nonzeros_in, nonzeros_out, in_strengths_L, out_strengths_L = calculate_nonzeros_and_strengths(L)

    fig, ax = plt.subplots(figsize=(12,6), ncols=2, nrows=1, dpi=150)

    ax[0].set_xlabel('In strength / Original in strength')
    ax[0].set_ylabel('Density')
    ax[0].set_yscale('log')
    ax[0].set_title('Ratio\'s In strength / Original in strength')

    ax[1].set_xlabel('Out strength / Original out strength')
    ax[1].set_ylabel('Density')
    ax[1].set_yscale('log')
    ax[1].set_title('Ratio\'s Out strength / Original out strength')

    for simulation in simulations.values():
        if L.nnz!=simulation.L.nnz:
            nonzeros_in, nonzeros_out, in_strengths_L, out_strengths_L = calculate_nonzeros_and_strengths(simulation.L)
        p_unweighted = simulation.p.copy()
        nonzeros = p_unweighted.data.nonzero()[0]
        p_unweighted.data[nonzeros] = np.ones(nonzeros.shape)

        in_strengths = np.array(simulation.p.sum(axis=0)).flatten()[nonzeros_in]
        out_strengths = np.array(simulation.p.sum(axis=1)).flatten()[nonzeros_out]

        ratio_in = in_strengths/in_strengths_L
        ax[0].hist(ratio_in, bins=10, histtype='step', density=density, label=f'{simulation.label}', linewidth=2)

        ratio_out = out_strengths/out_strengths_L
        ax[1].hist(ratio_out, bins=10, histtype='step', density=density, label=f'{simulation.label}', linewidth=2)

    ax[0].legend(loc='upper center')
    ax[1].legend(loc='upper center')

    plt.tight_layout()
    plt.show()
    
    
def show_capital(simulation):
    plt.figure(dpi=150, figsize=(8,5))

    plt.plot(simulation.total_reserves_history, label='Reserves')
    plt.plot(simulation.size_p, label='Payments')
    plt.plot(simulation.exo_history, label='Exogenous incoming')
    plt.plot(simulation.total_reserves_history+simulation.size_p, label='Payments+Reserves')

    plt.ylim(0,None)
    plt.xlim(0,None)

    plt.ylabel('EUR')
    plt.xlabel('Iteration')

    plt.title(f'Capital for simulation {simulation.label}')
    plt.legend()
    plt.tight_layout()
    plt.show()