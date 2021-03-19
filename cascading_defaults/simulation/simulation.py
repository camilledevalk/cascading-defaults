import numpy as np

from cascading_defaults.simulation.strategies import DefaultStrategy
from cascading_defaults.utils import save_contents, load_contents
from .. import plt  # This plt has nice settings :)

DTYPE = np.float64

class Simulation:
    available_strategies = ['EisenbergNoe', 'LargestCreditorLast', 'LargestCreditorFirst', 'LargestCreditorLastX',
                            'LargestCreditorFirstX', 'RobinHood', 'RobinHoodX', 'BlackHole', 'BlackHoleX']
        
    def __init__(self, strategy, L, label_of_run=None, exogenous_cashflow=np.zeros(0), start_reserves=0,
                 label_of_network='random_network', force_update_L=False):
        """
        Inputs:
        strategy: cascading_defaults.simulation.Strategy instance
        L: scipy sparse edgelist
        label: str, label for the simulation
        start_reserves: np.array with shape = (1,N) or float
        exogenous_cashflow: np.array with shape (1,N)
        """
        self.transaction_network = label_of_network

        # For saving
        self.skip_at_save = []
        self.skip_at_load = []
        
        assert isinstance(strategy, DefaultStrategy), f'Strategy {strategy} is of type ' \
                                                      f'{type(strategy)} and not of type {DefaultStrategy}.'
        
        # Creditors payment strategy
        self.strategy = strategy
        
        # Set label
        self.label_of_run = label_of_run
        label_of_run = f'_{label_of_run}' if self.label_of_run else ''
        self.label = f'{self.strategy.label}{label_of_run}'
        
        print(f'Setting up Simulation for {self.label}.')
        
        # Load right L
        self.L = self.strategy.select_right_L(L, label_of_network, force_update_L)
        
        self.N = self.L.shape[0]
        self.all_internal_nodes = np.array(list(set(np.array(self.L.nonzero()).flatten())))

        # Relative liabilities matrix
        self.relative_liabilities_matrix = None
        
        # Payment matrix
        self.p = None
        
        # Build reserves
        if isinstance(start_reserves, np.ndarray):
            self.reserves = start_reserves
        else:
            self.reserves = np.full(self.N, start_reserves)
        
        # Find p_i-bar (Equation (1) in Eisenberg and Noe [1])
        self.total_payables_vector = self.L.sum(axis=1)
        self.total_payables_array = np.array(self.total_payables_vector).flatten()
        
        # Total receivable cash
        self.total_receivables_vector = self.L.sum(axis=0)
        self.total_receivables_array = np.array(self.total_receivables_vector).flatten()
        
        # Exogenous
        if not self.strategy.has_exogenous:
            self.exogenous_cashflows = np.zeros(self.N)
        else:
            sum_payed_to_exogenous = self.total_receivables_array[0]
            # Downscaled to equal everything payed to exo
            self.exogenous_cashflows = exogenous_cashflow * (sum_payed_to_exogenous/exogenous_cashflow.sum())
        
        # Total equity of the firm is
        self.total_equities_array = np.max([np.zeros(self.N),
                                            self.exogenous_cashflows+self.total_receivables_array-
                                            self.total_payables_array],
                                            axis=0)
        
        # Set the starting values of receiving and payments:
        self.total_payments_array = np.zeros(self.N)
        self.total_receiving_array = self.total_receivables_array.copy()
        
        # Set starting 'reserves' (i.e. available money)
        self.reserves = self.reserves + self.total_receiving_array + self.exogenous_cashflows
        self.reserves[0] = 0  # Sinknode 0 doesn't take part in economy
        self.total_flow = self.reserves.sum()
        self.total_incoming = self.reserves + self.exogenous_cashflows
            
        # Let know whether has run or not
        self.has_run = False
        self.has_done_post = False
        self.loaded = False
        
    def _defaulting_nodes(self):
        """
        Returns an array of defaulting nodes (indices)
        """
        # A node is default when the obligations exceed the incoming cash
        defaulting = self.total_payables_array > self.total_payments_array
        
        # The indices
        self.nodes_currently_in_default = np.argwhere(defaulting)
        
        # Remove nodes that already defaulted in a previous stage
        new_defaults = np.setdiff1d(self.nodes_currently_in_default, self.defaulted_nodes)
        
        return new_defaults
    
    def _actual_run(self, rtol, max_iter=None, verbose=1):
        verboseprint = print if verbose else lambda *a, **k: None
        if self.loaded:
            self.has_run = True
            verboseprint('Not running, old files were loaded')
            return
        # Start the algorithm
        stage = 1
        terminate = False
        self.defaulted_nodes = []
        self.all_defaults = {}
        self.equities_history = []
        self.reserves_history = []
        self.total_reserves_history = []
        self.total_available_money_history = []
        self.exo_history = []
        
        np.random.seed(0)
        self.random_save_nodes = np.random.randint(0, self.N, size=100)
        
        # Total equities of all nodes
        self.equities = np.array(self.reserves - self.total_payables_array, dtype=DTYPE)
        self.equities_history.append(self.equities[self.random_save_nodes])
        
        while not terminate:   
            # Pay what you can (from reserves)
            ## Update reserves
            # Receive money
            # Update reserves
            # sum(reserves[-1]) == sum(reserves[0])
            
            # This is the quantity used in _payments_matrix()
            self.total_incoming = self.reserves # In this 'economy', you can pay from your reserves and exogenous
            
            # Calculate the payments p_ij
            self.p = self.strategy.payments_matrix(self) 
            
            # Calculate how much you pay
            total_payments = self.p.sum(axis=1)
            self.total_payments_array = np.array(total_payments).flatten()
            
            # Pay the money from your reserves
            if self.strategy.build_reserves:
                # Node 0 is ignored
                self.reserves[1:] = self.reserves[1:] - self.total_payments_array[1:]
                self.total_reserves_history.append(self.reserves[1:].sum())
            
            # 'Receive' money
            total_receiving = self.p.sum(axis=0)
            self.total_receiving_array = np.array(total_receiving).flatten()
            sum_payed_to_exogenous = self.total_receiving_array[0]
            
            # Decrease total exogenous available in a EisenbergNoe-ish way
            if self.strategy.has_exogenous:
                ratio = (sum_payed_to_exogenous/self.exogenous_cashflows.sum())
                self.exogenous_cashflows = self.exogenous_cashflows * ratio
                self.exo_history.append(self.exogenous_cashflows[1:].sum())
            
            # Add the received money to your reserves
            if self.strategy.build_reserves:
                # Node 0 is ignored
                self.reserves[1:] = self.reserves[1:] + self.total_receiving_array[1:] + self.exogenous_cashflows[1:]
                self.reserves_history.append(self.reserves[self.random_save_nodes])
                self.total_available_money_history.append(self.reserves[1:].sum())
            
            # Total equities of all nodes
            self.equities = np.array(self.total_incoming - self.total_payables_array, dtype=DTYPE)
            self.equities_history.append(self.equities[self.random_save_nodes])
            
            # A node is default when the obligations exceed the incoming cash
            self.defaults = self._defaulting_nodes()
            
            self.all_defaults[stage] = self.defaults
            self.defaulted_nodes.extend(self.defaults)
            
            # Display process
            sum_reserves = np.sum(self.reserves[1:])
            sum_payments = np.sum(self.total_payments_array[1:])
            verboseprint(
                f'\rstage: {stage:<4}, defaults: {len(self.defaults):<6}, sum reserves: {sum_reserves:1.2e}, '
                f'sum payments: {sum_payments:1.2e}, total flow: {100*sum_payments/sum_reserves:6.3f}%, sum payed to '
                f'exogenous: {sum_payed_to_exogenous:1.2e} sum exogenous cashflows: '
                f'{self.exogenous_cashflows.sum():1.2e}', end=''
            )
            
            # Wrap up the stage
            self.size_p.append(self.p.sum())
            self.size_p_relative.append(self.size_p[-1]/self.total_flow)
            
            # Terminate if the p vector doesn't change anymore
            if np.allclose(self.p.data, self.previous_p.data, rtol=rtol):
                terminate = True            
            
            stage += 1
            if stage > self.N:
                terminate = True
            if max_iter and stage >= max_iter:
                terminate = True
            self.previous_p.data = self.p.data
        del self.previous_ps_to_check
        self.has_run = True
        
        # Calculate how much you pay
        total_payments = self.p.sum(axis=1)
        self.total_payments_array = np.array(total_payments).flatten()

        # The difference of payments and receiving, you keep in your pockets
        if self.strategy.build_reserves:
            self.reserves = self.reserves - self.total_payments_array
            
        self.defaults = self._defaulting_nodes()
        
        return self
            
    def _post_run(self, save, verbose=1):
        verboseprint = print if verbose else lambda *a, **k: None
        if self.loaded:
            self.has_run = True
            verboseprint('Not post processing, old files were loaded')
            return
        
        # Wrap up simulation
        verboseprint('\n')

        all_nodes = np.array(list(set(np.array(self.L.nonzero()).flatten())))
        
        # Set stage 0 as all the never-defaulted nodes
        self.all_defaults[0] = np.setdiff1d(all_nodes, self.defaulted_nodes)
        verboseprint(f'defaulted: {len(self.defaulted_nodes)}')
        verboseprint(f'never defaulted: {len(self.all_defaults[0])}')
        
        # Create the default sequence
        self.default_sequence = np.zeros(len(self.all_defaults)-1)
        for i in range(1, len(self.all_defaults)):
            self.default_sequence[i-1] = len(self.all_defaults[i])
            
        # Converting lists to numpy
        self.reserves_history = np.array(self.reserves_history)
        self.total_available_money_history = np.array(self.total_available_money_history)
        self.equities_history = np.array(self.equities_history)
        self.exo_history = np.array(self.exo_history)
        self.size_p = np.array(self.size_p)
        self.size_p_relative = np.array(self.size_p_relative)
        
        # Save the object
        if save:
            self.save(verbose=verbose)
        
        self.has_done_post = True
        
    def run(self, rtol=5e-2, max_iter=None, save=False, verbose=1, actual_run=True):
        print(f'Running {self.label}.')
                
        # Clearing vector
        self.p = self.L.copy()  # Start with the assumption that it's just the network of obligations
        self.previous_p = self.p.copy()
        
        self.size_p = []
        self.size_p_relative = []
        self.previous_ps_to_check = None#[]
        
        if actual_run:
            self._actual_run(rtol, max_iter, verbose=verbose)
            self._post_run(save, verbose=verbose)
        
        print(f'Done with {self.label}.')
        
        return self            
    
    def show_example(self, stage=1, node=False):
        """
        This function allows the user to see a random node at a stage in the simulation.
        """
        if not node:
            random_defaulted = np.random.choice(self.all_defaults[stage])
            node = random_defaulted
            print(f'Random node from stage {stage}: {node}')
        else:
            print(f'Node {node}')
        print('\n\nLiabilities array:')
        print('\nPayables:')
        print(self.L.getrow(node))
        print(f'Total payables: {self.L.getrow(node).sum()}')
        print('\nReceivables:')
        print(self.L.getcol(node))
        print(f'Total receivables: {self.L.getcol(node).sum()}')
        print(f'Exogenous cashflow: {self.exogenous_cashflows[node]:.2f}')
        print(f'Total receiving: {self.L.getcol(node).sum() + self.exogenous_cashflows[node]:.2f}')
        print('\n\nClearing vector:')
        print('Outgoing:')
        print(self.p.getrow(node))
        print(f'Total Outgoing: {self.p.getrow(node).sum()}')
        print('\nIncoming:')
        print(self.p.getcol(node))
        print(f'Total receivables: {self.p.getcol(node).sum()}')
        print(f'Exogenous cashflow: {self.exogenous_cashflows[node]:.2f}')
        print(f'Total incoming: {self.p.getcol(node).sum() + self.exogenous_cashflows[node]:.2f}')
    
    def show_iterations(self):
        fig, ax1 = plt.subplots(figsize=(25,5))
        
        p_line = ax1.plot(self.size_p, label='$\|p^*\|$', color='g')
        L_line = ax1.plot(np.full(len(self.size_p), self.L.sum()), label='total flow', color='b')
        ax1.set_ylabel('total flow (EUR)')
        ax1.set_xlabel('iteration')
        ax1.set_ylim(0)
        
        lines = p_line + L_line
        labels = [l.get_label() for l in lines]
        
        ax2 = ax1.twinx()
        
        defaults_bar = ax2.bar(np.arange(len(self.size_p)), self.default_sequence,
                               color='g', alpha=0.2, label='defaults')
        ax2.set_yscale('log')
        ax2.set_ylabel('number of defaults')

        lines.append(defaults_bar)
        labels.append(defaults_bar.get_label())
        
        plt.legend(lines, labels)

        fig.tight_layout()
        
        self.clearing_vector_title = f'Clearing vector {self.label}'
        plt.title(self.clearing_vector_title)
        plt.show()
        
    def save(self, upperfolder=None, save_clearing_vector_as_nx=False, verbose=0, remove_existing=True):

        if not upperfolder:
            upperfolder = f'simulations/{self.transaction_network}/{self.label_of_run}/{self.strategy.label}'
 
        save_contents(self, upperfolder=upperfolder, verbose=verbose, label_seperate_folder=False,
                      remove_existing=remove_existing)
        
    def load(self, upperfolder=None, verbose=0, attributes=None):
        if not upperfolder:
            upperfolder = f'simulations/{self.transaction_network}/{self.label_of_run}/{self.strategy.label}'
        load_contents(self, upperfolder=upperfolder, verbose=verbose, label_seperate_folder=False,
                      attributes=attributes)
        
        self.loaded = True