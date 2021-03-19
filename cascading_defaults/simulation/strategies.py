import os

import numpy as np
from scipy import sparse

import cascading_defaults
from cython_defaults import cython_payments
from cython_sorting import sort_L_cython
from .. import current_dir


class DefaultStrategy:
    
    def __init__(self, build_reserves, pay_remaining_money, has_exogenous):
        # Characteristics
        if not hasattr(self, 'strategy'):
            self.strategy = None
        self.build_reserves = build_reserves
        self.pay_remaining_money = pay_remaining_money
        self.has_exogenous = has_exogenous
        
        # Label
        self.label = self.create_strategy_label()
    
    def create_strategy_label(self):
        """
        Function that creates a formatted label for the strategy, based on the parameters passed in __init__
        """
        X = 'X' if self.pay_remaining_money else ''
        R = 'R' if self.build_reserves else ''
        exo_label = 'Exo' if self.has_exogenous else 'NoExo'
        self.label = f'{self.strategy}{X}{R}-{exo_label}'
        return self.label
    
    def select_right_L(self, L, transaction_network='random_network', force_update_L=False, save_right_L=True):
        """
        Returns the right L
        """
        L_needed = self.L_needed()
        
        path_to_L = f'transactionnetworks/{transaction_network}/{L_needed}.npz'
        path_to_L = os.path.join(current_dir, path_to_L)
        safe_path_to_L = path_to_L.replace(os.getcwd(), '~')
        if (not force_update_L) and os.path.exists(path_to_L):
            print(f'Loading {L_needed} from {safe_path_to_L}.')
            with open(path_to_L, 'rb') as file:
                L = sparse.load_npz(file)
        else:
            print(f'Creating {L_needed}.')
            L = self.process_L_for_strategy(L)
            if save_right_L:
                try:
                    with open(path_to_L, 'wb') as file:
                        sparse.save_npz(file, L)
                except FileNotFoundError:
                    os.makedirs(os.path.dirname(path_to_L))
                    with open(path_to_L, 'wb') as file:
                        sparse.save_npz(file, L)
        
        return L
    
    def L_needed(self):
        if self.has_exogenous:
            return 'L_sinknode'
        else:
            return 'L'
    
    def process_L_for_strategy(self, L):
        return L

    def payments_matrix(self, simulation):
        if not hasattr(self, 'simulation_checked'):
            assert isinstance(simulation, cascading_defaults.simulation.simulation.Simulation), f'Simulation {simulation} is of type {type(simulation)} and not of type {cascading_defaults.simulation.simulation.Simulation}.'
            self.simulation_checked = True
    
    
class EisenbergNoe(DefaultStrategy):
    
    def __init__(self, build_reserves, pay_remaining_money, has_exogenous):
        self.strategy = 'EisenbergNoe'
        assert pay_remaining_money == True, f'Strategy EisenbergNoe works by definition with pay_remaining_money=True, you passed: {pay_remaining_money}.'
        super().__init__(build_reserves, pay_remaining_money, has_exogenous)
        
    def L_needed(self):
        if self.has_exogenous:
            return 'L_sinknode_EisenbergNoe'
        else:
            return 'L_EisenbergNoe'
        
    def process_L_for_strategy(self, L):
        # Find p_i-bar (Equation (1) in Eisenberg and Noe [1])
        total_payables_vector = L.sum(axis=1)
        
        # Total receivable cash
        total_receivables_vector = L.sum(axis=0)
        
        # Correct for zero liabilities
        L[np.argwhere(total_payables_vector==0), np.argwhere(total_payables_vector==0)] = 0
        
        return L
        
    def payments_matrix(self, simulation):
        super().payments_matrix(simulation)
        
        if not hasattr(self, 'relative_liabilities_matrix'):
            self.relative_liabilities_matrix = self.init_relative_liabilities_matrix(simulation)
        
        total_dollar_payments = np.minimum(simulation.total_incoming, simulation.total_payables_array).reshape((-1,1))
        # Calculate new payment matrix
        # This is the incoming cash divided over all the nodes it has an obligation to
        # Multiply is an element-wise (column) multiplication
        return self.relative_liabilities_matrix.multiply(total_dollar_payments)
            
    def init_relative_liabilities_matrix(self, simulation):
        ones = np.ones((simulation.L.shape[0],1))
        
        multiplier = np.divide(ones, simulation.total_payables_vector, out=np.zeros_like(ones), where=simulation.total_payables_vector != 0)
        
        self.relative_liabilities_matrix = simulation.L.multiply(multiplier).tocsr()  # Note that multiply is column-wise
        
        # Correct for zero liabilities
        self.relative_liabilities_matrix[np.argwhere(simulation.total_payables_vector==0), np.argwhere(simulation.total_payables_vector==0)] = 1
        
        return self.relative_liabilities_matrix

class LargestCreditor(DefaultStrategy):
    
    def L_needed(self):
        if self.last_first == 'first':
            self.ascending_descending = 'descending'
        elif self.last_first == 'last':
            self.ascending_descending = 'ascending'
        else:
            raise Exception(f'{self.last_first} not a valid ordering')
        if self.has_exogenous:
            return f'L_sinknode_sorted_{self.ascending_descending}'
        else:
            return f'L_sorted_{self.ascending_descending}'
        
    def process_L_for_strategy(self, L):
        L_sorted = sort_L_cython(L, ascending_descending=self.ascending_descending)
        
        return L_sorted
    
    def payments_matrix(self, simulation):
        super().payments_matrix(simulation)
        
        return cython_payments(simulation, strategy='largest_creditor', last_first=self.last_first, pay_remaining_money=self.pay_remaining_money)
    
    
class LargestCreditorFirst(LargestCreditor):
    
    def __init__(self, build_reserves, pay_remaining_money, has_exogenous):
        self.strategy = 'LargestCreditorFirst'
        self.last_first = 'first'
        super().__init__(build_reserves, pay_remaining_money, has_exogenous)
        
class LargestCreditorLast(LargestCreditor):
    
    def __init__(self, build_reserves, pay_remaining_money, has_exogenous):
        self.strategy = 'LargestCreditorLast'
        self.last_first = 'last'
        super().__init__(build_reserves, pay_remaining_money, has_exogenous)
        
        
        
available_strategies = [LargestCreditorFirst, LargestCreditorLast, EisenbergNoe]