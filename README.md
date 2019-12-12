# RBIG for Spatial-Temporal Exploration of Earth Science Data

This repo has the experiments and reproducible code for all of my experiments using RBIG for analyzing Earth science data. I am primarily focused on using RBIG to measure the information content for datasets with spatial-temporal features. I look at IT measures such as Entropy, Mutual Information and Total Correlation. The strength of RBIG lies in it's ability to handle multivariate and high dimensional datasets.

I will be periodically updating this repo as I finish more experiments.


---
### Example Experiments

I have included some example experiments in the notebooks folder including the following experiments:

* Global Information Content (TODO)
* Spatial-Temporal Analysis of variables 
* Temporal analysis of Drought Indicators
* Climate Model Comparisons


---
### Installation Instructions

You can create the environment from the `.yml` file found in the main repo.

```python
conda env create -f environment.yml -n myenv
source activate myenv
```




---
### Conferences

* Estimating Information in Earth Data Cubes - Johnson et. al. - EGU 2018
* Multivariate Gaussianization in Earth and Climate Sciences - Johnson et. al. - Climate Informatics 2019 - [repo](https://github.com/IPL-UV/2019_ci_rbig)
* Climate Model Intercomparison with Multivariate Information Theoretic Measures - Johnson et. al. - AGU 2019 - [slides](https://docs.google.com/presentation/d/18KfmAbaJI49EycNule8vvfR1dd0W6VDwrGCXjzZe5oE/edit?usp=sharing)


### Journal Articles

* Iterative Gaussianization: from ICA to Random Rotations - Laparra et. al. (2011) - IEEE Transactions on Neural Networks
* Information theory measures and RBIG for Spatial-Temporal Data analysis - Johnson et. al. - In progress

---
### External Toolboxes

**RBIG (Rotation-Based Iterative Gaussianization)**

This is a package I created to implement the RBIG algorithm. This is a multivariate Gaussianization method that allows one to calculate information theoretic measures such as entropy, total correlation and mutual information. More information can be found in the repository [`esdc_tools`](https://github.com/IPL-UV/py_rbig).

**Earth Science Data Cube Tools**

These are a collection of useful scripts when dealing with datacubes (datasets in xarray `Dataset` format). I used a few preprocessing methods as well as a `Minicuber` implementation which transforms the data into spatial and/or temporal features. More information can be found in the repository [`py_rbig`](https://github.com/IPL-UV/esdc_tools).

---
### Data Resources

**Earth System Data Lab**

This is sponsered by the [earthsystemdatalab](https://www.earthsystemdatalab.net/). They host a cube on their servers which include over 40+ variables including soil moisture and land surface temperature. They also feature a free-to-use [JupyterHub server](https://www.earthsystemdatalab.net/index.php/interact/data-lab/) for easy exploration of th data. 

**Climate Data Store**

This is a database of climate models implemented by the ECMWF and sponsored by the Copernicus program. I use a few climate models from here by using the CDSAPI. There are some [additional instructions](https://cds.climate.copernicus.eu/api-how-to) to install this package which requires registration and agreeing to some terms of use for each dataset.

---
## Contact Information

Links to my co-authors' and my information:

* J. Emmanuel Johnson - [Website](https://jejjohnson.netlify.com) | [Scholar](https://scholar.google.com/citations?user=h-wdX7gAAAAJ&hl=es) | [Twitter](https://twitter.com/jejjohnson) | [Github](https://github.com/jejjohnson) 
* Maria Piles - [Website](https://sites.google.com/site/mariapiles/) | [Scholar](https://scholar.google.com/citations?hl=es&user=KTva-HMAAAAJ) | [Twitter](https://twitter.com/Maria_Piles)
* Valero Laparra - [Website](https://www.uv.es/lapeva/) | [Scholar](https://scholar.google.com/citations?user=dNt_xikAAAAJ&hl=es)
* Gustau Camps-Valls - [Website](https://www.uv.es/gcamps/) | [Scholar](https://scholar.google.com/citations?user=6mgnauMAAAAJ&hl=es) | [Twitter](https://twitter.com/isp_uv_es)