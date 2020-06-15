# <div align="center">Noah's Data Science Portfolio</div>

<br>

## 1. House Sale Price Model
Skills Demonstrated: *multi-variate analysis, regression, multi-collinearity testing, exploratory data analysis*<br />
Libraries: *pandas, matplot, seaborn*

### Project Overview
Using a competition dataset from Kaggle with over 75 explanatory variables, I created a model that predicts the sale price of a house in Ames, Iowa based on its description. The purpose of this project was to demonstrate my ability to use an exploratory data analysis on a large, complicated dataset to produce a reliable predictive model.

### Analysis

### Results

### Resources
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

<br>

## 2. Avian Abundance and Diversity Project 
Skills Demonstrated: *data sourcing, data wrangling, data cleaning, exploratory data analysis*<br />
Libraries: *pandas, numpy, matplot, seaborn*

### Project Overview
In the Fall of 2019, a colleague and I launched the Monitoring of Beneficial Birds in Agricultural Ecosystems initiative. The purpose of this project is to connect sustainable land use practices with changes in native bird communities. We are particularly interested in species that provide ecosystem services or are of conservation concern. To date, we have completed two survey seasons (Fall 2019 and Spring 2020) at five organic farms in Ford County, Illinois. Our current goal is to secure an additional 5 years of funding ($10,000) to support seasonal surveys and an undergraduate researcher at the University of Illinois.

**Dallas Glazik - dglazik@gmail.com**<br />
Outreach, communication, grants coordinator<br />
**Noah Horsley - nphorsley59@gmail.com**<br />
Surveyor, analysist, project coordinator<br />
**Colin Dobson - cdobson2@illinois.edu**<br />
Surveyor, undergraduate researcher

### Enrolled Farms
Cow Creek (400 acres)<br />
Craver Trust (200 acres)<br />
D & Q (80 acres)<br />
J & W (160 acres)<br />
R Wildflower & Fields (160 acres)

### Analysis

### Results 

### Resources
https://www.birdpop.org/docs/misc/Alpha_codes_eng.pdf

<br>

## 3. Partial Life-cycle Matrix Model
Skills Demonstrated: *data simulation, predictive modeling, problem solving*<br />
Libraries: *pandas, numpy, matplot, copy*<br />
Filename: PLC_MatrixModel.py

### <div align="center">Project Overview</div>
As part of my Master's degree, I built a partial life-cycle matrix model and used it to predict the survival rate of Common Grackles (blackbird) during the non-breeding season. The model simulates realistic population growth and has a wide range of additional functionalities that extend beyond the scope of my thesis, including a flexible parameter design, built-in elasticity analysis capabilities, stable age distribution estimates, and much more. I am in the process of publishing the results produced by this model. I am also using this model to do predictive modeling for another publication investigating Micronesian Starling demography on the island of Guam.

#### Stage 1
The first stage simulates population growth over a pre-defined trial period to transform a single population variable into an array representing stable age distribution. 

#### Stage 2
The second stage proportionally adjusts the stable age distribution array to represent the desired initial population size. From here, population growth is simulated over the length of time defined by the user.

#### Stage 3
The third and final stage averages annual population size and growth over the number of simulations defined by the user. It then plots mean population size with confidence intervals and calculates an annual trend estimate.

#### Additional Functionality
The three stages outlined above describe the most simplistic modeling approach. Additional details and feature descriptions can be found in the PLC_MatrixModel_AdvFunc.txt file included in this repository.

### <div align="center">Results</div>
**Table 1.** 
![alt text](https://github.com/nphorsley59/Portfolio/blob/master/Pop_Matrix_Table1.png "Age Matrix")
**Table 2.** 
![alt text](https://github.com/nphorsley59/Portfolio/blob/master/Pop_Growth_Table1.png "Population Growth")

