# Optimization Tools 0.1.0

This package provides tools for solving generic multi-objective, constrained optimization problems. Specifically, it contains classes such as `Optimizer` and `Cost` that can be used to easily implement various types of general-purpose optimization schemes such as simulated annealing and genetic algorithms, using arbitrary cost functions as well as arbitrary 'hard' constraints.

The repo was designed with certain graph-based costs (e.g. distance metrics), and related applications such as sequencing (traveling salesperson-like problems) and clustering (k-medoids, etc) in mind, and so contains powerful defaults to handle these cases. 

This repo was spun out of [Sequencing-POC](https://github.com/VersesTech/sequencing-poc) and for now retains some extraneous code relating to specific applications, as well as for loading graph data from specific servers, for ease of testing. 

---

## *Setup*

This package requires `networkx`, `numpy`, and `pandas` as well as a few other common dependencies. If you need to install any of these you can run

```bash
pip install -r requirements.txt
```

Some of the demos in this repo depend on graphs constructed from waypoint and bin location data stored on a remote server. Data will be downloaded from the server as needed, but this can be time-consuming. To cache local copies of all domain graphs for which data is available, simply run:

```bash
python setup_graphs.py
```

---

## *Usage examples*

### Simulated annealing demo

A simple example that runs simulated annealing to find an efficient route on a random graph, while visiting a specified list of nodes, can be run as follows: 

```bash
python -m demos.optimizer_demo
```
The resulting optimized node sequence and distance should be displayed in the terminal, and a visualization of the resulting path displayed in a pop-up window.

This code also demonstrates the syntax for constructing a basic instance of the `Optimizer` class, described below.

---

## *Classes*

This is a breakdown of the main classes included in the repo and the intended function of each. See docstrings for full documentation.

- `opt.model.optimizer.Optimizer` - This is the main class that the repo is organized around. An optimizer that minimizes one or more cost functions (supplied as a list of `CostFn` instances) given optional constraints (supplied as `ConstraintFn` instances). Defaults to a standard implementation of simulated annealing (e.g. a random walk using mutations from genetic algorithms and the Metropolis criterion with an annealed temperature parameter).

- `opt.model.cost.CostFn` - Constructs a class corresponding to a cost function, given a `cost_fn_handle` (a Python function used to compute the cost), and optional `args`, `kwargs`, and encoder function (`encoder_fn_handle`), which encodes the input to the optimizer in whatever form the cost function expects. Dedicated args and kwargs can be supplied for the encoder function as well, which defaults to an identity transform.

- `opt.model.energy.EnergyFn` - Constructed internally by the `Optimizer` class -- combines multiple cost functions along with their weights to return a single cost for the optimizer to minimize.

- `opt.model.constraint.ConstraintFn` - Uses a `constraint_fn_handle` to supply constraints on optimization problems that are used to check the output of an Optimizer. The constraint function should return Boolean values and may operate directly on the state being optimized or (via an optional argument) on a cost that is a function of the state.

- `opt.model.modifier.ModifierFn` - Wrapper class around one or more `mod_fn_handles` that specify 'step functions' determining how states are modified to produce new candidates for evaluation (i.e. to take a step in a Markov chain). If more than one step function is supplied, step functions drawn from a categorical distribution (supplied as `mod_probs`, defaulting to a flat distribution) at each iteration of optimization.

---

## *Graphs*

Code for working with graphs (including random graphs as well as 'domain graphs' representing digital twins of particular warehouses) has been ported from the Sequencing POC. Please see documentation there for details, as well as `opt.graph` in the present repo.

---

## *Demos*

To run various demos, run

```bash
python -m demos.demo_name
```

---

## *Tests*

To run the unit tests included with this repo, simply run

```bash
pytest
```

in the repo root directory (assuming `pytest` is installed). All tests should be passed with very high probability (since some algorithms involve stochastic elements, there is a very small chance that a solution found for one or two of the tests will not be optimal and will cause a test to fail.)
