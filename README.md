# UNAGI

UNAGI: Deep Generative Model for Deciphering Cellular Dynamics and In-Silico Drug Discovery in Complex Diseases

## Overview 
<img title="UNAGI Overview" alt="Alt text" src="UNAGI_overview.png">
UNAGI is a comprehensive unsupervised in-silico cellular dynamics and drug discovery framework. UNAGI adeptly deciphers cellular dynamics from human disease time-series single-cell data and facilitates in-silico drug perturbations to earmark therapeutic targets and drugs potentially active against complex human diseases. All outputs, from cellular dynamics to drug perturbations, are rendered in an interactive visual format within the UNAGI framework. Nestled within a deep learning architecture Variational Autoencoder-Generative adversarial network (VAE-GAN), UNAGI is tailored to manage diverse data distributions frequently arising post-normalization. It also innovatively employs disease-informed cell embeddings, harnessing crucial gene markers derived from the disease dataset. On achieving cell embeddings, UNAGI fabricates a graph that chronologically links cell clusters across disease stages, subsequently deducing the gene regulatory network orchestrating these connections. UNAGI is primed to leverage time-series data, enabling a precise portrayal of cellular dynamics and a superior capture of disease markers and regulators. Lastly, the deep generative prowess of UNAGI framework powers an in-silico drug perturbation module, simulating drug impacts by manipulating the latent space informed by real drug perturbation data from the CMAP database. This allows for an empirical assessment of drug efficacy based on cellular shifts towards healthier states following drug treatment. The in-silico perturbation module can similarly be utilized to investigate therapeutic pathways, employing an approach akin to the one used in drug perturbation analysis.

## Key Capabilities
-   Learning disease-specific cell embeddings through iterative training processes.

-   Constructing temporal dynamic graphs from time-series single-cell data and reconstructing temporal gene regulatory networks to decipher cellular dynamics.

-   Identifying dynamic and hierarchical static markers to profile cellular dynamics, both longitudinally and at specific time points.

-   Performing *in-silico* perturbations to identify potential therapeutic pathways and drug/compound candidates.

## Installation

UNAGI installation
```
git clone https://github.com/mcgilldinglab/UNAGI.git
cd UNAGI
python setup.py install 
```

### Prerequisites
-   Python >=3.9 (Python3.9 is recommanded)
-   pyro-ppl>=1.8.6
-   scanpy>=1.9.5
-   anndata==0.8.0
-   torch >= 2.0.0
-   matplotlib>=3.7.1

**iDREM installation:**

```
git clone https://github.com/phoenixding/idrem.git
```

**iDREM prerequisites:**

Install the iDREM to the source folder of UNAGI

-   Java
    To use iDREM, a version of Java 1.7 or later must be installed. If Java 1.7 or later is not currently installed, please refer to http://www.java.com for installation instructions.

-   JavaScript
    To enable the interactive visualization powered by Javascript, please make sure that you have Internet connection.
    (The users are still able to run the software off-line, but Internet access is needed to view the result interactively.)


## Tutorials:

### Dataset preparation

[Prepare datasets to run UNAGI.](https://github.com/mcgilldinglab/UNAGI/blob/main/tutorials/dataset_preparation.ipynb)

### Training and analysis on example dataset

[UNAGI training and analysis on an example dataset.](https://github.com/mcgilldinglab/UNAGI/blob/main/tutorials/run_UNAGI_using_example_dataset.ipynb)

### Visualize the results of UNAGI method

[Visualization on an example dataset.](https://github.com/mcgilldinglab/UNAGI/blob/main/tutorials/visualize_UNAGI_results_example_dataset.ipynb)

## Contact
[Yumin Zheng](mailto:yumin.zheng@mail.mcgill.ca)
