Last updated: 09.2022

Overview
--------

This repository is associated with the paper "Inferring social influence in animal groups across multiple timescales". 

Many animal behaviours exhibit complex temporal dynamics, suggesting there are multiple timescales at which they should be studied. Despite this, researchers have mostly focussed on isolating behaviours that occur over relatively restricted temporal scales, typically corresponding to those most accessible to human observation. The situation becomes more complex still when we consider multiple animals interacting, where behavioural coupling can introduce further, new timescales of importance. We analysed two species that move in different media—golden shiner fish and homing pigeons and show that different interpretations regarding social influence can arise from analysing behaviour at different timescales, highlighting the importance of explicitly considering its multi-scale nature.

Here, we provide all code required to reproduce the results presented in this paper.

Analysis
---------

The analysis pipeline is split into two parallel implementations, one for golden shiner fish and the other for homing pigeons. The naming convention of all analyses files encodes the species and the order in which code needs to be run in the prefix '00X_', where '00' is the number that denotes the order in which the code needs to be run and 'X' denotes the speces (F: fish and B: birds). Note that parts of code where the prefix is limited to '00' without 'X' represents code that is common for both species. For example, '03_DirectionalCorrelation.cu'. 

All code except the calculation of time-lagged directional correlations was written using [Python](https://www.python.org/) and [jupyter notebooks](https://jupyter.org/). This specific code was written in [CUDA](https://developer.nvidia.com/cuda-toolkit).
