# Experiments

In this document, I outline some of the experiments


## Experiment I

For the first experiment, I would like to see if there is a difference between each of the methods and what interactions are they able to capture.

**Independent Variables**

* Regions
    * [ ] Europe
    * [ ] Spain
    * [ ] USA
    * [ ] Africa
* Time Periods
    * [ ] July 2010
    * [ ] Jan 2010
* Variables:
    * [ ] Gross Primary Productivity (GPP)
    * [ ] Root Soil Moisture (RSM)
    * [ ] Land Surface Temperature (LST)
* Methods:
    * [ ] pV Coefficient - (Norm, pV score)
    * [ ] CKA w. RBF Kernel - (K Norm, CKA score)
    * [ ] RBIG Method - (H, I)

---

**Dependent Variables**

For the dependent variables, we want to see how the spatial and temporal features change the outcome of the scores that we calculate. We will vary the amount of spatial dimensions 

* Spatial v Temporal Dimensions
    * [ ] Spatial Dims - (1, 2, 3, 4, 5)
    * [ ] Temporal Dims - (1, 2, 3, 4, 5)

---

**Hypothesis**

I'm assuming that the PDF estimation with RBIG will be superior. We are adding a large amount of dimension/features to the inputs. I suspect that this can be potentially useful up to a certain point. Due to the amount of information included, some methods will be able to capture the changes while other methods no. Some things to keep in mind about the methods:

* RBIG is designed for multivariate/multidimensional data.
* pV Coefficient is a linear method which cannot capture non-linear relationships
* CKA is a non-linear method but it may not capture the relationships due to the parameter estimation



