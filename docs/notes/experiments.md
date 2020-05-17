---
title: Experiments
description: The experiments that need to be conducted for the RBIG 4 EO paper
authors:
    - J. Emmanuel Johnson
path: docs/notes
source: experiments.md
---
# Experiments

In this document, I outline some of the experiments that we will be conducting in the RBIG 4 EO paper.

---

## Experiment I - Toy Examples

---

## Experiment II - Spatial-Temporal

For the first experiment, I would like to see what relationships we can see with information and entropy between different spatial-temporal relationships. We will take different variables and use RBIG as a density estimator and as an information theory estimator.

### Independent Variables

* Regions
  * [ ] Europe
  * [ ] Spain
  * [ ] World
* Time Periods
  * [ ] July 2010
  * [ ] Jan 2010
* Variables:
  * [ ] Gross Primary Productivity (GPP)
  * [ ] Root-Zone Soil Moisture (RSM)
  * [ ] Precipitation (era5)
  * [ ] Land Surface Temperature (LST)
  * [ ] Leaf Area Index (LAI)
* Methods:
  * [ ] pV Coefficient - (Norm, pV score)
  * [ ] CKA w. RBF Kernel - (K Norm, CKA score)
  * [ ] RBIG Method - (H, I)

---

### Dependent Variables

For the dependent variables, we want to see how the spatial and temporal features change the outcome of the scores that we calculate. We will vary the amount of spatial dimensions.

* Spatial v Temporal Dimensions
  * [ ] Spatial Dims - (1, 2, 3, 4, 5)
  * [ ] Temporal Dims - (1, 2, 3, 4, 5)

---

### Preprocessing

I will do some basic preprocessing strategies to ensure that we get a somewhat accurate representation of what we want.

#### Climatology

I'll remove the climatology by taking the mean of each month and then subtract that from each pixel within that month.

#### Resampling

I will do a basic resampling strategy which takes the mean of every month. I will be a rolling mean.

---

### Hypothesis

I'm assuming that the PDF estimation with RBIG will be superior. We are adding a large amount of dimension/features to the inputs. I suspect that this can be potentially useful up to a certain point. Due to the amount of information included, some methods will be able to capture the changes while other methods no. Some things to keep in mind about the methods:

* RBIG is designed for multivariate/multidimensional data.
* pV Coefficient is a linear method which cannot capture non-linear relationships
* CKA is a non-linear method but it may not capture the relationships due to the parameter estimation
