# OreoNSpy

Optical resonator numerical simulation in Python.

Current package version: 4.4.2

## Requirements

- Python >= 3.10
- numpy >= 1.23
- matplotlib >= 3.5
- Optional acceleration: numba >= 0.60

## Installation

```bash
git clone https://github.com/matib12/oreonspy.git
cd oreonspy
```

### Inside conda evironment

```bash
conda env create --file=environment.yml --name=oreonspy
```

### Using pip

Install from this repository:

```bash
pip install .
```

Install in editable mode for development:

```bash
pip install -e .
```

Install optional numba support:

```bash
pip install .[numba]
```

Build a wheel:

```bash
python -m build
```

## Quick Start

```python
import numpy as np
import oreonspy as op

# Build a cavity
cavity = op.Cavity(t_a=0.1, r_a=0.9, r_b=0.9, cavity_length=3000.0)

# Initialize simulation
lambd = 1064e-9
requested_sampling_frequency = 1450e3
initial_input_electric_field = 1.0 + 0.0j
cavity.simulation(
	lambd,
	requested_sampling_frequency,
	initial_input_electric_field,
	backend="auto",  # "auto" | "pure" | "numba"
)

cavity.print_sim_params()

# One simulation step
intracavity_field, reflected_field = cavity.sim_step(
	input_electric_field=1.0 + 0.0j,
	input_mirror_displacement=0.0,
	output_mirror_displacement=0.0,
)
```

## Examples

The repository includes practical notebook examples in the `nb` folder.

- Main tutorial: `nb/example.ipynb`
- Sampling and simulator checks: `nb/Resampling test.ipynb`
- Notebook for accompanying paper: `nb/Reproduce figure 2 from Lee at al 1999.ipynb`

We suggest to start from the `nb/example.ipynb`

## Public API

- Cavity
- HAS_NUMBA

## Repository Structure

- nb: Jupyter notebooks
- src: Python package source code
- docs: notes and simulation strategy material

## Project Links

- repository GitHub: https://github.com/matib12/oreonspy
- description paper arXiv: 
