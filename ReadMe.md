

## Introduction

This is the source code of the paper entitled [**An Evolutionary Multi-Objective Neural Architecture Search Approach to Advancing Cognitive Diagnosis in Intelligent Education**](https://ieeexplore.ieee.org/abstract/document/10599558/). 
If this is also the code of [*Designing Novel Cognitive Diagnosis Models via Evolutionary Multi-Objective Neural Architecture Search*] (https://arxiv.org/abs/2307.04429). You can also find the old version at [here](https://github.com/DevilYangS/EMO-NAS-CD/tree/main/Old%20Version).




## Dependency Install
To start with, you may install some packages to meet the dependency as 
1. Create your environment by conda
   ```
   conda create envname python=3.9
   ```
2.  install dependency by
   ```
pip install -r requirements.txt
   ```
**or** manually install the following packages
```
pytorch>=1.4.0
scikit-learn
pandas
networkx # used for plotting neural architectures
```


## Prepare Datasets
- Process your target dataset or Download datasets from [EduData](https://github.com/bigdata-ustc/EduData). This repository provides five datasets, including ASSIST09 (termed assist), SLP, ASSIST12, ASSIST2017, and Junyi. 
- 

## Usage
```
# Search process
python EvolutionSearch.py

# Training process
python main.py
