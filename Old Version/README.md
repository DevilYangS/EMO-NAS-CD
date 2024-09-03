
## Notes

This is the old version of the code for EMO-NAS-CD ([for arxiv version](https://arxiv.org/pdf/2307.04429)), but it does not contain the code of comparison approaches.
However, it is still effective to search cognitive diagnosis models, its effectiveness is same as that of new project.



## Introduction

This is the source code of the paper entitled [*Designing Novel Cognitive Diagnosis Models via Evolutionary Multi-Objective Neural Architecture Search*](https://arxiv.org/pdf/2307.04429).

More descriptions about the found models and usages can be found in the [new project](https://github.com/DevilYangS/EMO-NAS-CD)




## Dependency Install
```
pytorch>=1.4.0
scikit-learn
pandas
networkx # used for plotting neural architectures
```

## Usage
```
# Search process
python EvolutionSearch.py

# Training process
python main.py
