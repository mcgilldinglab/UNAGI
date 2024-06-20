# Usage

(installation)=
## Installation

Create a new conda environment

```bash
  $ conda create -n unagi python=3.9
  $ conda activate unagi
```

### UNAGI installation

**Option 1: Install from pip**

```bash
  $ pip install scUNAGI
```

**Option 2: Install from Github**

Installing UNAGI directly from GitHub ensures you have the latest version.
```bash
  $ git clone https://github.com/mcgilldinglab/UNAGI.git
  $ cd UNAGI
  $ pip install .
```

### Prerequisites
-   Python >=3.9 (Python3.9 is recommanded)
-   pyro-ppl>=1.8.6
-   scanpy>=1.9.5
-   **anndata==0.8.0** 
-   torch >= 2.0.0
-   matplotlib>=3.7.1
-   iDREM (Mandatory)

**iDREM installation:**

```
git clone https://github.com/phoenixding/idrem.git
```

**iDREM prerequisites:**

-   Java
    To use iDREM, a version of Java 1.7 or later must be installed. If Java 1.7 or later is not currently installed, please refer to http://www.java.com for installation instructions.

-   JavaScript
    To enable the interactive visualization powered by Javascript, please make sure that you have Internet connection.
    (The users are still able to run the software off-line, but Internet access is needed to view the result interactively.)

## Example data

To use the example dataset, run the tutorials or reproduce the manuscript results.

-  clone the github repo.

``` bash
  $ git clone https://github.com/mcgilldinglab/UNAGI.git
```

-   Download required files.

**Required files**

Preprocessed CMAP database: [One Drive.](https://mcgill-my.sharepoint.com/:u:/g/personal/yumin_zheng_mail_mcgill_ca/EazTbqa3vKtJnwd6-DL87GUBaAwEA8AXaHHCdEXtS1bPFg?e=Y5A2WO)
-    **Mandatory** data to run UNAGI.

Preprocessed IPF snRNA-seq dataset: [One Drive.](https://mcgill-my.sharepoint.com/:f:/g/personal/yumin_zheng_mail_mcgill_ca/EhUPO3Ip0IhCh0kz-Uply_MBzksNoX9N6HDEgC_dUHbCkg?e=biVLuV)
-    UNAGI outcomes to reproduce the figures and tables generated for the manuscript.

Example dataset: [Link.](https://github.com/mcgilldinglab/UNAGI/tree/main/UNAGI/data/example)
-   The dataset for UNAGI walktrhough demonstration. 

