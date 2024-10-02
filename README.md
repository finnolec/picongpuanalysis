# PIConGPU analysis

## Introduction

This library is a collection of analysis scripts for my own PIConGPU simulations and it is mainly supposed to stream-line my ever-growing jupyter notebooks into a more maintainable state. 
The simulations are focusing on LWFAs/PWFAs and synthetic diagnostics of these accelerators.

## Installation

Create conda environment
```bash
mamba create --name picongpuanalysis-python3.12 python=3.12 jupyter ipython matplotlib
mamba activate picongpuanalysis-python3.12
```

Install poetry
```bash
pipx install poetry
```

Download library
```bash
git clone git@github.com:finnolec/picongpuanalysis.git
cd picongpuanalysis
```

Maybe required to update lock file:
```bash
poetry lock
```

Install picongpuanalysis with poetry
```bash
poetry install --all-extras
```

## Examples

You can find examples on how to use different modules of this library in [this directory](examples/).
