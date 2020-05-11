# Results for Droughts

## Recap

Recall that we are trying to investigate how a new variable called Vegetation Optical Depth (VOD) compares to some of the previously used variables Land Surface Temperature (LST), Soil Moisture (SM) and Normalized Vegetation Difference Index (NDVI). The previous variables were often used to recover droughts, but studies have shown that VOD will better characterize drought conditions. So, the objective is to see how these VOD compares to other variables via information theory and other similarity measures. In addition, the data representation could make a difference in how similar VOD is to the other variables. For example, many spatial dimension or many temporal dimensions. So, we are going to look at different representations and different combinations of VOD, LST, SM, and NDVI and do comparisons.

---

#### Data

 For this first set of results, we are looking at the CONUS dataset which is located in California. We have a period of 6 years where there were drought occurences and not drought occurences.

* Droughts - 2012, 2014, 2015)
* No Drought - 2010, 2011, 2013

We have a spatial resolution of ... and a temporal resolution of 14 days.

---

#### Methods

**Entropy**

For this first experiment, we only want to classify the expected uncertainty that each variable has with different temporal representations. So for example, we can do self-comparisons like what's the expected uncertainty of VOD with 1 temporal dimension versus 3 temporal dimensions? We will do this for all 4 variables individually. We will use RBIG to measure the expected uncertainty (aka Entropy)

$$
H(\mathbf{X}) = \mathbb{E}_\mathbf{x} \left[ -\log p(\mathbf{x}) \right]
$$

**Similarity**

For this second experiment, we want to see how **similar** VOD is to the other variables under different spatial-temporal representations. E.g. VOD vs. SM, VOD vs. NDVI, etc. So there are a few ways to do this. We want to show that we can do it with Mutual Information (MI) because it uses probability distributions:

$$
I(\mathbf{X,Y}) = H(\mathbf{X}) + H(\mathbf{Y}) - 2 H(\mathbf{X,Y})
$$

We also have some of the standard measures like

* Pearson correlation coefficient
* Spearman correlation coefficient
* RV-Coefficient (Multivariate extension of pearson)
* centered kernel alignment (CKA)

---

#### Hypothesis

Adding temporal features will definitely increase the amount of entropy within a variable. We also expect to see some differences in the entropy value obtained from the different variables. However, there should be a point where adding more temporal dimensions may not add much more information.

> We see some trend that perhaps gives us intuition that there could be a 'sweet' spot for the amount of temporal dimensions to use.

---

#### Preprocessing



1. **Climatology**

I remove the climatology because we want to characterize the anomalies outside of the climate patterns.

2. **Similar Data**

I ensure that the lat-lon-time locations are consistent across variables. 

**Note**: We will have less samples as we increase the number of features because of the boundaries.

---

## Entropy

For the entropy, 
=== "Drought"
    ![png](../pics/droughts/lines/H_individual_drought.png)

=== "Non-Drought"
    ![png](../pics/droughts/lines/H_individual_nondrought.png)

=== "Both"
    ![png](../pics/droughts/lines/H_individual_both.png)


## Mutual Information

??? details "VOD"
    === "Drought"

        === "MI"
            ![png](../pics/droughts/lines/VOD_I_norm_drought.png)

        === "Pearson"
            ![png](../pics/droughts/lines/VOD_rv_coef_drought.png)

        === "Kernels"
            ![png](../pics/droughts/lines/VOD_cka_coeff_drought.png)

    === "Not-Drought"

        === "MI"
            ![png](../pics/droughts/lines/VOD_I_norm_nondrought.png)

        === "Pearson"
            ![png](../pics/droughts/lines/VOD_rv_coef_nondrought.png)

        === "Kernels"
            ![png](../pics/droughts/lines/VOD_cka_coeff_nondrought.png)

    === "Both"

        === "MI"
            ![png](../pics/droughts/lines/VOD_I_norm_both.png)

        === "Pearson"
            ![png](../pics/droughts/lines/VOD_rv_coef_both.png)

        === "Kernels"
            ![png](../pics/droughts/lines/VOD_cka_coeff_both.png)

??? details "NDVI"
    === "Drought"

        === "MI"
            ![png](../pics/droughts/lines/NDVI_I_norm_drought.png)

        === "Pearson"
            ![png](../pics/droughts/lines/NDVI_rv_coef_drought.png)

        === "Kernels"
            ![png](../pics/droughts/lines/NDVI_cka_coeff_drought.png)

    === "Not-Drought"

        === "MI"
            ![png](../pics/droughts/lines/NDVI_I_norm_nondrought.png)

        === "Pearson"
            ![png](../pics/droughts/lines/NDVI_rv_coef_nondrought.png)

        === "Kernels"
            ![png](../pics/droughts/lines/NDVI_cka_coeff_nondrought.png)

    === "Both"

        === "MI"
            ![png](../pics/droughts/lines/NDVI_I_norm_both.png)

        === "Pearson"
            ![png](../pics/droughts/lines/NDVI_rv_coef_both.png)

        === "Kernels"
            ![png](../pics/droughts/lines/NDVI_cka_coeff_both.png)

??? details "SM"
    === "Drought"

        === "MI"
            ![png](../pics/droughts/lines/SM_I_norm_drought.png)

        === "Pearson"
            ![png](../pics/droughts/lines/SM_rv_coef_drought.png)

        === "Kernels"
            ![png](../pics/droughts/lines/SM_cka_coeff_drought.png)

    === "Not-Drought"

        === "MI"
            ![png](../pics/droughts/lines/SM_I_norm_nondrought.png)

        === "Pearson"
            ![png](../pics/droughts/lines/SM_rv_coef_nondrought.png)

        === "Kernels"
            ![png](../pics/droughts/lines/SM_cka_coeff_nondrought.png)

    === "Both"

        === "MI"
            ![png](../pics/droughts/lines/SM_I_norm_both.png)

        === "Pearson"
            ![png](../pics/droughts/lines/SM_rv_coef_both.png)

        === "Kernels"
            ![png](../pics/droughts/lines/SM_cka_coeff_both.png)

??? details "LST"
    === "Drought"

        === "MI"
            ![png](../pics/droughts/lines/LST_I_norm_drought.png)

        === "Pearson"
            ![png](../pics/droughts/lines/LST_rv_coef_drought.png)

        === "Kernels"
            ![png](../pics/droughts/lines/LST_cka_coeff_drought.png)

    === "Not-Drought"

        === "MI"
            ![png](../pics/droughts/lines/LST_I_norm_nondrought.png)

        === "Pearson"
            ![png](../pics/droughts/lines/LST_rv_coef_nondrought.png)

        === "Kernels"
            ![png](../pics/droughts/lines/LST_cka_coeff_nondrought.png)

    === "Both"

        === "MI"
            ![png](../pics/droughts/lines/LST_I_norm_both.png)

        === "Pearson"
            ![png](../pics/droughts/lines/LST_rv_coef_both.png)

        === "Kernels"
            ![png](../pics/droughts/lines/LST_cka_coeff_both.png)

### Taylor Diagram

??? details "Temporal Dims: 1"

    === "Both"
        ![png](../pics/droughts/taylor/vod_both_1.png)

    === "Drought"
        ![png](../pics/droughts/taylor/vod_True_1.png)

    === "Non-Drought"
        ![png](../pics/droughts/taylor/vod_False_1.png)


??? details "Temporal Dims: 2"

    === "Both"
        ![png](../pics/droughts/taylor/vod_both_2.png)

    === "Drought"
        ![png](../pics/droughts/taylor/vod_True_2.png)

    === "Non-Drought"
        ![png](../pics/droughts/taylor/vod_False_2.png)

??? details "Temporal Dims: 3"

    === "Both"
        ![png](../pics/droughts/taylor/vod_both_3.png)

    === "Drought"
        ![png](../pics/droughts/taylor/vod_True_3.png)

    === "Non-Drought"
        ![png](../pics/droughts/taylor/vod_False_3.png)

??? details "Temporal Dims: 4"

    === "Both"
        ![png](../pics/droughts/taylor/vod_both_4.png)

    === "Drought"
        ![png](../pics/droughts/taylor/vod_True_4.png)

    === "Non-Drought"
        ![png](../pics/droughts/taylor/vod_False_4.png)

??? details "Temporal Dims: 5"

    === "Both"
        ![png](../pics/droughts/taylor/vod_both_5.png)

    === "Drought"
        ![png](../pics/droughts/taylor/vod_True_5.png)

    === "Non-Drought"
        ![png](../pics/droughts/taylor/vod_False_5.png)

??? details "Temporal Dims: 6"

    === "Both"
        ![png](../pics/droughts/taylor/vod_both_6.png)

    === "Drought"
        ![png](../pics/droughts/taylor/vod_True_6.png)

    === "Non-Drought"
        ![png](../pics/droughts/taylor/vod_False_6.png)

??? details "Temporal Dims: 7"

    === "Both"
        ![png](../pics/droughts/taylor/vod_both_7.png)

    === "Drought"
        ![png](../pics/droughts/taylor/vod_True_7.png)

    === "Non-Drought"
        ![png](../pics/droughts/taylor/vod_False_7.png)

??? details "Temporal Dims: 8"

    === "Both"
        ![png](../pics/droughts/taylor/vod_both_8.png)

    === "Drought"
        ![png](../pics/droughts/taylor/vod_True_8.png)

    === "Non-Drought"
        ![png](../pics/droughts/taylor/vod_False_8.png)

??? details "Temporal Dims: 9"

    === "Both"
        ![png](../pics/droughts/taylor/vod_both_9.png)

    === "Drought"
        ![png](../pics/droughts/taylor/vod_True_9.png)

    === "Non-Drought"
        ![png](../pics/droughts/taylor/vod_False_9.png)

??? details "Temporal Dims: 10"

    === "Both"
        ![png](../pics/droughts/taylor/vod_both_10.png)

    === "Drought"
        ![png](../pics/droughts/taylor/vod_True_10.png)

    === "Non-Drought"
        ![png](../pics/droughts/taylor/vod_False_10.png)

??? details "Temporal Dims: 11"

    === "Both"
        ![png](../pics/droughts/taylor/vod_both_11.png)

    === "Drought"
        ![png](../pics/droughts/taylor/vod_True_11.png)

    === "Non-Drought"
        ![png](../pics/droughts/taylor/vod_False_11.png)

??? details "Temporal Dims: 12"

    === "Both"
        ![png](../pics/droughts/taylor/vod_both_12.png)

    === "Drought"
        ![png](../pics/droughts/taylor/vod_True_12.png)

    === "Non-Drought"
        ![png](../pics/droughts/taylor/vod_False_12.png)