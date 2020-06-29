# <div align="center">Noah's Data Science Portfolio</div>

<br>

## 1. House Sale Price Model
Skills Demonstrated: *multi-variate analysis, regression, multi-collinearity testing, exploratory data analysis*<br />
Libraries and Programs: *pandas, matplotlib, seaborn*

### Project Overview
Using a competition dataset from Kaggle with over 75 explanatory variables, I created a model that predicts the sale price of a house in Ames, Iowa based on its description. The purpose of this project was to demonstrate my ability to use an exploratory data analysis on a large, complicated dataset to produce a reliable predictive model.

### Analysis

### Results

### Resources
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

<br>

## 2. Avian Abundance and Diversity Project
Skills Demonstrated: *data sourcing, data wrangling, data cleaning, lambda functions, exploratory data analysis*<br />
Libraries and Programs: *Python, Spyder, Tableau, pandas, numpy*<br />
Filename(s): AAD_DataCleaning.py, AAD_SpeciesCodes.py

### <div align="center">Project Overview</div>
In 2019, a colleague and I launched the Monitoring of Beneficial Birds in Agricultural Ecosystems Initiative. The purpose of this project is to connect sustainable land use practices with changes in native bird communities. As the sole analyst, I am responsible for transforming raw data into a format that can be quickly and easily communicated with landowners and funding agencies. For the Spring 2020 dataset, I used standard data wrangling, data cleaning, and data visualization techniques to create an interactive report.

### <div align="center">Data Wrangling</div>

Avian count data is collected and entered as 4-letter "Alpha" codes. While these codes are meaningful to ornithologists, they do a poor job of communicating study results to the general public. I decided I'd need to present full species names when reporting data for this project. I used Python to turn this table (Figure 1) published by The Institute for Bird Populations<sup>2</sup> into a Python dictionary (Figure 2). I then used the dictionary to connect 4-letter "Alpha" codes in my dataset to the full English species names.<br />

**Figure 1.** A small sample of the over 2,100 bird species that have been assigned 4-letter "Alpha" codes.<br />

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/AAD_Figures/Bird_Species_Codes_Table1.png "Alpha Codes to English Names Table")<br />

**Figure 2.** The same sample after being transformed into a Python dictionary.<br />

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/AAD_Figures/Bird_Species_Codes_Table2.png "Alpha Codes to English Names Dictionary")<br />

### <div align="center">Data Cleaning</div>

After establishing an "Alpha" codes reference dictionary, I began cleaning the Monitoring of Beneficial Birds in Agricultural Ecosystems Initiative dataset. I find data cleaning to be most effecient and thorough when divided into phases. For this project, I used the following approach:

#### Phase 1 - Identification
The first phase was to identify general problems with the dataset. I used .dtypes and a .value_counts() loop to create a fast summary of each column. I then used this summary to list out obvious tasks (Figure 3). While this was a good start, I had not addressed the possibility of NaNs in the dataset. To view NaNs, I used .isna().sum().sort_values(ascending=False) to view NaNs by column (Figure 4). Again, I listed out any obvious cleaning tasks.

**Figure 3.** An organized approach to cleaning data.<br /> 

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/AAD_Figures/Data_Cleaning_Table1.png "Data Cleaning Tasks")<br />

**Figure 4.** A simple method for summarizing NaNs in a dataset.<br /> 

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/AAD_Figures/Data_Cleaning_Table2.1.png "Table of NaNs by Column")<br />

#### Phase 2 - Cleaning
The second phase was to complete tasks identified in Phase 1. I used common indexing functions, such as .loc/iloc and .at/iat, to identify and address typos and other minor errors. More widespread problems were addressed using more aggressive functions and techniques, such as .replace(), .fillna(), lambda functions, loops, and custom functions (Figure 5).

**Figure 5.** A loop used to move data that had been entered into the wrong column.<br />

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/AAD_Figures/Data_Cleaning_Table3.png "Moving Data with a Loop")<br />

#### Phase 3 - Quality Assurance
The third phase was to repeat Phase 1 and, if necessary, Phase 2 to ensure nothing was missed in the initial cleaning process. In this particular project, I was unable to link English species names to the "Alpha" codes in my dataset until some obvious errors had been fixed i.e. until after Phases 1 and 2. However, after linking the English species names to the "Alpha" codes, it quickly became clear that errors existed in the "Alpha" codes column (Figure 6). These errors were difficult to catch in Phases 1 and 2 because they existed in a diverse categorical variable with no 'reference' set available for verification. I find this second round of cleaning, which I call "Quality Assurance", to be most useful in large or error-prone datasets.

**Figure 6.** Identifying rows with "Alpha" code (SpeciesCode column) errors.<br />

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/AAD_Figures/Data_Cleaning_Table4.png "Alpha Code Errors")<br />

#### Phase 4 - Usability
The final phase was to increase the usability and readability of the dataset. A "clean" dataset that is difficult to understand/interpret is not very useful for analysis. For this dataset, I cleaned up uneccesary codes, renamed some columns, reordered the columns, and transformed some discrete data (e.g. StartTime) into continuous data. The final product was a clean, organized, easy-to-read dataset that was ready for analysis (Figure 7).

**Figure 7.** A cleaned sample from the Spring 2020 dataset.<br />

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/AAD_Figures/Data_Cleaning_Table5.1.png "Cleaned Dataset")<br />

### <div align="center">Visualization Using Tableau</div>

I used Tableau to summarize the results of our Spring 2020 surveys. I have included a sample plot below (Figure 8). The full workbook<sup>3</sup> can be found on Tableau Public.<br />

**Figure 8.** A bar chart showing prominent members of the bird community (>10 individuals) at each site.

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/AAD_Figures/BirdCbS_Sp2020_Table1.png "Bird Community by Site")<br />

### <div align="center">Resources</div>
<sup>1</sup> https://www.cowcreekorganics.com/about<br />
<sup>2</sup> https://www.birdpop.org/docs/misc/Alpha_codes_eng.pdf<br />
<sup>3</sup> https://public.tableau.com/profile/noah.horsley#!/vizhome/ILOrganicFarmSurveysSP2020/RWildflowerFarmandFieldsLLC_1<br />


<br>

## 3. Partial Life-cycle Matrix Model
Skills Demonstrated: *data simulation, predictive modeling, custom functions, feature engineering*<br />
Libraries and Programs: *Python, Jupyter Notebook, pandas, numpy, matplotlib, math, copy*<br />
Filename(s): PLC_MatrixModel.ipynb

### <div align="center">Project Overview</div>
As part of my Master's thesis, I built a partial life-cycle matrix model<sup>1</sup> and used it to predict the survival rate of Common Grackles<sup>2</sup> during the non-breeding season. The model simulates realistic population growth and has several additional features that extend beyond the scope of my research. I am in the process of publishing the results produced by this model. I am also using this model to do predictive modeling for another publication investigating Micronesian Starling demography on the island of Guam.

### <div align="center">Model Structure</div>

#### Stage 1
The first modeling stage simulates population growth over a pre-defined trial period to transform a single population variable into an array representing stable age distribution. 

#### Stage 2
The second modeling stage proportionally adjusts the stable age distribution array to represent the desired initial population size. From here, population growth is simulated over the length of time defined by the user.

#### Stage 3
The third and final modeling stage averages annual population size and growth over the number of simulations defined by the user. It then plots mean population size with confidence intervals and calculates an annual trend estimate.

#### Additional Features
The three stages outlined above describe the most simplistic modeling approach. More details can be found in the PLC_MatrixModel_AddFeat.txt file included in my Portfolio repository.

### <div align="center">Results</div>

Matrix models use matrices to track population growth across age classes (Table 1). This model structure allowed me to input age-specific survival and fecundity for more accurate modeling results. 

**Table 1.** A sample of the age matrix dataframe that shows population distribution across age classes.<br />

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/PLC_MatrixModel_Figures/Pop_Matrix_Table1.png "Age Matrix")<br />

In order to account for environmental stochasticity, I built natural variation into each rate based on real data collected in the field. As a result, each simulation produces slightly different results (Table 2, Figure 2). 

**Table 2.** A sample of the population growth dataframe that show population growth across simulations.<br />

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/PLC_MatrixModel_Figures/Pop_Growth_Table1.png "Population Growth")<br />

For my thesis, we used established demographic rates from the literature and estimated demographic rates from our own research (Table 3) to predict non-breeding season survival in three distinct populations: a stable population, the current global population, and the current Illinois population (Table 4, Figure 1). 

**Table 3.** The model parameters used to predict non-breeding season survival.<br />

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/PLC_MatrixModel_Figures/Model_Parameters_Table1.png "Model Parameters")<br />

**Table 4.** The predicted rates of non-breeding season survival for each population.<br />

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/PLC_MatrixModel_Figures/NBS_Survival_Predictions_Table1.png "Model Predictions")<br />

**Figure 1.** Population growth projections using predictions of non-breeding survival for three distinct populations.<br />
&nbsp;  

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/PLC_MatrixModel_Figures/Proj_Pop_Growth_Figure1.png "Predicted Population Growth")<br />

**Figure 2.** A sample of simulations produced by the model, replicating projected decline in Illinois.<br />
![alt text](https://github.com/nphorsley59/Portfolio/blob/master/PLC_MatrixModel_Figures/livesim_plot_24sims.gif "Simulation Animation")
### <div align="center">Summary</div>

The partial life-cycle matrix model I built for my Master's thesis was used to project population decline in my study species, the Common Grackle, and to predict rates of non-breeding season survival for three distinct populations: a stable population, the current global population, and the current Illinois population. The modeling approach I chose allowed me to use many demographic parameters and account for realistic environmental variation. I learned a lot about model design, custom functions, and advanced visualization techniques from this project and am excited to reuse the model to answer other research questions in the future.

### <div align="center">Resources</div>
<sup>1</sup> https://onlinelibrary.wiley.com/doi/abs/10.1034/j.1600-0706.2001.930303.x<br />
<sup>2</sup> https://birdsoftheworld.org/bow/species/comgra/cur/introduction<br />
