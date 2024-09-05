
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

You can directly use the found models by EMO-NAS-CD on ASSISTments2009 and SLP datasets. Their searching logs can be found at `./experiment/AssistmentSearch_22-07-08_18-11-39/` and `./experiment/slpSearch_22-08-16_23-07-02`, respectively. 

The non-dominated individuals/models can be extracted from  `Gen_99/fitness.txt` and `Gen_99/Population.txt`  as follows:
```
from EMO_public import NDsort
import numpy as np

fitness =np.loadtext(`Gen_99/fitness.txt`)
Popsize=fitness.shape[0]
FrontValue = NDsort.NDSort(fitness, Popsize)[0]

nondominated_individuals = np.where(FrontValue==0)[0]+1

```
The Pareto front and seven selected models (on two datasets) are  shown as follows

<img src='images/ASSIST.png' alt="sym" width="50%"> <img src='images/SLP.png' alt="sym" width="50%">

Here *A1-A7* were found on the  **ASSISTments2009 dataset**, *S1-S7* were found on the  **SLP dataset**. 
For *A1-A7*, their `nondominated_individuals` is **[2,37,15,25,98,73,82]**, while for *S1-S7*, their `nondominated_individuals` is [3,6,29,18,12,41,57].
In short, A1_A7= [2,37,15,25,98,73,82], S1_S7=[3,6,29,18,12,41,57]. 

With  the `nondominated_individuals` index, you can extract their decision variables from `Gen_99/Population.txt`.

To save these models conveniently, each model is saved with integer encoding. 
That is, the integer encoding （three-bit for illustrating one node）, where the first and second bits denote that the node receives which input, and the third bit specifies that the node adopts which operation in ``Models/Operations``.
For example, [0,1,10] denotes the node takes the *student* and 'exercise' as inputs and uses the `Add` operation.  

Their integer encoding is as follows:
```
DECSPACE = []
# A1-A7
DECSPACE.append( [0, 1, 10] )
DECSPACE.append([0, 2, 10, 1, 2, 11, 3, 4, 10] )
DECSPACE.append([1, 2, 10, 3, 0, 6, 0, 2, 11, 0, 5, 11, 4, 6, 10] )
DECSPACE.append([0, 1, 10, 0, 2, 10, 3, 4, 11, 5, 0, 13] )
DECSPACE.append([1, 2, 10, 0, 2, 10, 3, 4, 11, 5, 0, 13, 6, 0, 6] )
DECSPACE.append([2, 0, 3, 1, 3, 10, 0, 4, 11, 5, 0, 13, 6, 0, 6] )
DECSPACE.append( [2, 0, 1, 1, 3, 10, 0, 4, 11, 5, 0, 13, 6, 0, 0, 7, 0, 6] )

# S1-S7
DECSPACE.append([0, 1, 10]  )
DECSPACE.append(  [0, 1, 10, 3, 0, 7] )
DECSPACE.append(  [0, 1, 10, 3, 0, 2, 0, 0, 3, 4, 5, 10] )
DECSPACE.append( [1, 0, 13, 0, 0, 13, 3, 4, 10, 5, 0, 6] )
DECSPACE.append( [0, 1, 10, 2, 3, 11, 4, 0, 13, 5, 0, 6] )
DECSPACE.append( [1, 1, 12, 1, 3, 11, 0, 0, 9, 5, 2, 11, 4, 6, 10, 7, 0, 6, 8, 0, 13] )
DECSPACE.append( [1, 1, 12, 1, 3, 11, 0, 0, 9, 5, 2, 11, 4, 6, 10, 0, 0, 9, 8, 0, 0, 9, 0, 4, 7, 10, 10, 11, 0, 6, 12, 0, 13] )
```



## Usage
```
# Search process
python EvolutionSearch.py

# Training process
python main.py
