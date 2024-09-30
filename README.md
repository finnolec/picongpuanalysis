Create conda environment
```
mamba create --name picongpuanalysis-python3.12 python=3.12 jupyter ipython matplotlib
mamba activate picongpuanalysis-python3.12
```

Install poetry
```
pipx install poetry
```

Maybe required to update lock file:
```
poetry lock
```

Install picongpuanalysis with poetry
```
poetry install --all-extras
```
