# tesi_svizzeretto


## Repository structure

* nb - jupyter notebooks
* optics - Finesse simulations
* src - python module source files (oreonspy)
* dist - python module installation files (oreonspy)
* OresoNS - C++ optical reasonator dynamics simulator

# OreoNSpy
## Install
To install the module in your environment as editable use wheel file:
```
conda activate your_environment
pip install oreonspy-[version]-py3-none-any.whl --no-deps
```

To generate the wheel file issue the following command in the project directory:
```
python3 -m build
```

In some cases use:
```
pip install --editable .
```

## Use
Create your cavity
```
import oreonspy

my_cavity = oreonspy.Cavity(t_a = 0.1, r_a = 0.9, r_b = 0.9, L=3000.)
```

Define the light interacting with the cavity
```
lambd = 1064e-9  # m
k = 2.*np.pi / lambd
```

Initialize the simulation and make steps
```
my_cavity.simulation(k, 1450e3)
my_cavity.print_sim_params()

my_cavity.sim_step(d_zeta, E_in_curr=1.)
```

## Install Gymnasium
Install dependencies manually (via conda preferebly) if necessary.

Then install Gymnasium in editable way:

```
git clone https://github.com/Farama-Foundation/Gymnasium.git
cd Gymnasium/
pip install --editable .
```
