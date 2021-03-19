# Casdading defaults
This is the repo of my graduation research on cascading defaults.

## Research methods
### `1-simulation.ipynb`
In this notebook, the simulation (fictitious default algorithm) can be run.

First, some parameters can be set. Then the `Simulation` objects are initialised.

One can choose to run all simulations in parallel (multiprocessing), though, this is quite unstable with finishing.

Smoother is to run the simulations in sequence.

Saving is done by setting `save=True` in the `.run()` method.

### `2-analysing-simulations.ipynb`
First, it is selected which simulations to analyse.

Then, the different figures are generated using the module `cascading_defaults.analysis.simulation_analysis`.