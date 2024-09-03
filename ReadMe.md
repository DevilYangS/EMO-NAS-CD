
# EMO-NSA-CD

## Introduction

This is the source code of the paper entitled [**An Evolutionary Multi-Objective Neural Architecture Search Approach to Advancing Cognitive Diagnosis in Intelligent Education**](https://ieeexplore.ieee.org/abstract/document/10599558/). 
If this is also the code of [*Designing Novel Cognitive Diagnosis Models via Evolutionary Multi-Objective Neural Architecture Search*](https://arxiv.org/abs/2307.04429). You can also find the old version at [here](https://github.com/DevilYangS/EMO-NAS-CD/tree/main/Old%20Version).




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
- Current dir has contained three datasets: ASSIST09, SLP, and Junyi. Complete datasets can be found at the [release](https://github.com/DevilYangS/EMO-NAS-CD/releases/tag/datasets).
- Then you can run the *divide_data.py* to get the variant datasets under different splitting ratios as
  ```
   divide_data(name='slp',ratio=0.6) # ratio from 0.4 to 0.7
  ```
  That will generate the train/validation/test datasets for SLP under the ratio of 0.6/0.1/0.3.
  You can generate others by setting ratio to from 0.4 to 0.7 for ASSIST09 and Junyi.

- For ASSIST12 and ASSIST17, only the datasets under ratio of 0.7 is needed, but you can also run the *Re_divide_data_for_12_2017.py* to re-divide the dataset as
  ```
   redivide_data(name='assist12',ratio=0.6) 
  ```
   That will generate the train/validation/test datasets for ASSIST12 under the ratio of 0.6/0.1/0.3.


## Directly Use the Found Models by EMO-NSA-CD and Others

## Usage
```
# Search process
python EvolutionSearch.py

# Training process
python main.py
