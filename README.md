# Understanding Transformer Memorization Recall Through Idioms
This repository contains a dataset of idioms for probing memorization in pretrained LMs. "[Understanding Transformer Memorization Recall Through Idioms](https://arxiv.org/abs/2210.03588)".


## Datasets
IdioMem is a probe dataset of English idioms used to analyze memorization behavior in language models (e.g., GPT and BERT were examined in the paper). 

The dataset is formatted as a JSONL file and can be found [here](https://github.com/adihaviv/idiomem/blob/main/idiomem.jsonl).

Details of the construction of IdioMem and the methodical approach for probing via IdioMem are described in the paper. 

## Code
- IdiomMem dataset construction can be found under idioms_dataset_collector.py (it uses the source data files located under "data" folder).
- To run all experiments from the paper, you can use "run_experiments.py" script. This script will create a pkl file for the requested model and dataset. 


## Citation

If you find this work helpful, please cite us
```
@inproceedings{haviv2023understanding,
  title={Understanding Transformer Memorization Recall Through Idioms},
  author={Haviv, Adi and Cohen, Ido and Gidron, Jacob and Schuster, Roei and Goldberg, Yoav and Geva, Mor},
  booktitle={Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
  pages={248--264},
  year={2023}
}
```

This repo is WIP and will contain all the code and analysis for the paper. For any questions and requests, please reach out to me at adi.haviv@cs.tau.ac.il
