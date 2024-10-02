# Examples

## Shadowgraphy

[shadowgraphy.ipynb](shadowgraphy.ipynb) loads both the intermediate and final output from the PIConGPU shadowgraphy plugin. The intermediate output is then processed with the `picongpuanalysis.postprocessing` module to reproduce the final output. Both outputs are then compared to each other. Generally, the `picongpuanalysis.postprocessing` module can be used to create shadowgrams more flexibly then the in-situ version.
