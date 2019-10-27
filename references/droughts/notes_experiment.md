# Notes

## Demo Experiment 

So in this experiment, I will looking at how the variables interact with one another depending upon the amount of temporal features we include. I will look at IT measures as well as some standard correlation measures to see how these effects change with the amount of features used.

**Summary**

* I will be changing the amount of temporal features 

#### Variable Parameters

* Temporal Range, 14 days to 6 months
* Variables
  * Normalized Difference Vegetation Index (NDVI)
  * Land Surface Temperature (LST)
  * Soil Moisture (SM)
  * Vegetation Optical Depth (VOD)

#### Fixed Parameters

* Region - California
* Density Estimator - RBIG Algorithm
* Scores
  * Standard - Pearson, Spearman
  * IT - Entropy, Total Correlation, Mutual Information


#### Steps

1. Load Cube Data
2. Get Region (California)
3. Get Variables 
4. Get Density Cubes
5. Get Density Estimator
6. Get IT scores
7.  Save Results

#### Code Notes

**Classes**

* DataLoader
* DataSaver
* BuildFeature
* Model
* Metrics
* Experiment
