# Dynamic Correlation Clustering in Sublinear Update Time

This repository contains the code and Jupyter Notebooks required to reproduce the experiments from the paper **"Dynamic Correlation Clustering in Sublinear Update Time"**.

## Installation

To run the notebooks, you will need to have Python installed along with a few key libraries. The required libraries are:

- `csv`
- `random`
- `utils`
- `collections`
- `time`
- `matplotlib`
- `seaborn`
- `numpy`

You can install the required libraries using pip:

```sh
pip install matplotlib seaborn numpy
```

Note: `csv`, `random`, `collections`, and `time` are part of Python's standard library and do not need to be installed separately.

## Notebooks

The following Jupyter Notebooks are included to reproduce different experiments:

- `Comparison with classical Pivot.ipynb`: Used to create Table 3 and assumes that the input graph is contained in a file where each line contains two space separated node ids representing an edge.  
- `Embedding graphs - camera ready.ipynb`: Used to create Table 2 and assumes that the input is read from a txt file where every line corresponds to a the embedding vector for a point.
- `Final plot creation.ipynb`: Used to create Figures 1-4 and Table 1. Assumes the same input graph format as the first notebook.

Each notebook contains detailed instructions and explanations of the experiments being conducted.

---
