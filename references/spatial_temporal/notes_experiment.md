# Notes

## Demo Experiment 

So in this experiment I will be looking at how the spatial-temporal features can change the expected information content of a model. We will be looking at it from two perspectives: the modeling perspective where we can access the sensitivity measures and the IT perspective where we can simply measure the information content.

* I will be changing the amount of spatial and temporal features 
* I will vary the regions 
* I will use two different ML methods to observe
* I will use 

#### Variable Parameters

* Temporal Features - 8 days to 6 months
* Spatial Features - 1 grid space to 7 grid spaces
* Regions - Europe, Africa, USA
* Variables
  * Gross Primary Productivity (GPP)
  * Land Surface Temperature (LST)
  * Soil Moisture (SM)
  * Root Moisture (RM)

#### Fixed Parameters

* Density Estimator - RBIG Algorithm
* IT Scoring Methods - IT measures, ML Scores, Sensitivity Measures
* ML Algorithm - Gaussian Process
* ML Scoring Metrics - R2, MSE, MAE, RMSE


#### Steps

1. Load Cube Data
2. Get Region
3. Get Variables
4. Get Density Cubes
5. Get Density Estimator
6. Get IT scores
7. Get ML Model
8. Get ML Scores
9. Get ML Sensitivity (Gradient)
10. Save Results

#### Code Notes

**Classes**

* DataLoader
* DataSaver
* Model
* Metrics
* Experiment
