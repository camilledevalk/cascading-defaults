from itertools import product
import os
import pandas as pd
import pickle
from .. import current_dir
import shutil

def save_contents(custom_object, upperfolder=None, label=None, remove_existing=True, skip_at_save=[], verbose=0, label_seperate_folder=True):
    """
    Saves all non-trivial attributes from an object.
    skip_at_save can be a list of names of attributes to skip at saving.
    """
    verboseprint = print if verbose else lambda *a, **k: None
    
    if hasattr(custom_object, 'label'):
        label = custom_object.label
    assert label, 'No valid label for the object'
    
    if upperfolder:
        if label_seperate_folder:
            path = os.path.join(current_dir, upperfolder, label)
        else:
            path = os.path.join(current_dir, upperfolder)
    else:
        path = os.path.join(current_dir, label)
        
    if os.path.exists(path):
        existing_files = os.listdir(path)
        if remove_existing:
            shutil.rmtree(path)
    else:
        existing_files = []

    safe_path = path.replace(os.getcwd(), '~')
    print(f'Saving to {safe_path}/')
        
    all_attributes = custom_object.__dir__()
    if hasattr(custom_object, 'skip_at_save'):
        for attribute in getattr(custom_object, 'skip_at_save'):
            skip_at_save.append(attribute)
    for attribute in all_attributes:
        if (attribute in skip_at_save) or (attribute in existing_files and (not remove_existing)):
            continue
        if 'method' not in str(type(getattr(custom_object, attribute))) and '__' not in attribute:
            filename = os.path.join(path, attribute)
            if type(getattr(custom_object, attribute)) == pd.DataFrame:
                filename = filename + '.df.pkl'
                verboseprint(f'Saving {filename}')
                try:
                    with open(filename, 'wb') as file:
                        getattr(custom_object, attribute).to_pickle(file, compression=None)
                except FileNotFoundError:
                    os.makedirs(os.path.dirname(filename))
                    with open(filename, 'wb') as file:
                        getattr(custom_object, attribute).to_pickle(file, compression=None)
            else:
                filename = filename + '.pkl'
                verboseprint(f'Saving {filename}')
                try:
                    with open(filename, 'wb') as file:
                        pickle.dump(getattr(custom_object, attribute), file)
                except FileNotFoundError:
                    os.makedirs(os.path.dirname(filename))
                    with open(filename, 'wb') as file:
                        pickle.dump(getattr(custom_object, attribute), file)
    # Also save its type
    filename = os.path.join(path, 'type.pkl')
    verboseprint(f'Saving {filename}')
    with open(filename, 'wb') as file:
        type_of_object = str(type(custom_object))
        pickle.dump(type_of_object, file)
        
        
def load_contents(empty_object, upperfolder=None, label=None, skip_at_load=[], verbose=0, label_seperate_folder=True, attributes=None):
    """
    Loads all data from a folder into the 'empty object'
    """
    skip_at_load.append('type')
    verboseprint = print if verbose else lambda *a, **k: None
    
    if hasattr(empty_object, 'label'):
        label = empty_object.label
    assert label or (not label_seperate_folder), 'No valid label for the object'
    
    if upperfolder:
        if label_seperate_folder:
            path = os.path.join(current_dir, upperfolder, label)
        else:
            path = os.path.join(current_dir, upperfolder)
    else:
        path = os.path.join(current_dir, label)
        
    # Check if the type matches what is stored in the folder
    filename = os.path.join(path, 'type.pkl')
    with open(filename, 'rb') as file:
        type_of_object = pickle.load(file)
        
    assert type_of_object == str(type(empty_object)), f'Type of saved object ({type_of_object}) does not equal type of passed object ({str(type(empty_object))}).'

    safe_path = path.replace(os.getcwd(), '~')
    print(f'Loading from {safe_path}/')
        
    files_in_path = [os.path.join(path, f) for f in os.listdir(path)]
    
    if not attributes:
        attributes = [os.path.split(f)[1].split('.')[0] for f in files_in_path]
    else:
        files_in_path_old = files_in_path
        files_in_path = []
        for file in files_in_path_old:
            if os.path.split(file)[1].split('.')[0] in attributes:
                files_in_path.append(file)
    
    if 'skip_at_load' in attributes:
        file = files_in_path[attributes.index('skip_at_load')]
        verboseprint(f'Loading {file}')
        with open(file, 'rb') as f:
            skip_at_load_ = pickle.load(f)
        skip_at_load.extend(skip_at_load_)
    for file, attribute in zip(files_in_path, attributes):
        if attribute in skip_at_load:
            continue
        # Check for empty file 
        if file == '':
            continue
        else:
            verboseprint(f'Loading {file}')
            with open(file, 'rb') as f:
                setattr(empty_object, attribute, pickle.load(f))


def select_simulations(simulations, selection):
    """
    Creates a subset of the dictionary where the simulation is in the selection.
    Inputs:
    simulations, dict of simulation
    selection, tuple of features of the simulation, like (strategy_classes, build_reserves_list, pay_remaining_money_list, has_exogenous_list)
    Outputs:
    A subset of the dictionary
    """
    if not selection:
        return simulations
    selection = list(product(*selection))
    selected = {}
    
    for key, simulation in zip(simulations, simulations.values()):
        strategy = simulation.strategy
        features = (type(strategy), strategy.build_reserves, strategy.pay_remaining_money, strategy.has_exogenous)
        if features in selection:
            selected[key] = simulation
    return selected