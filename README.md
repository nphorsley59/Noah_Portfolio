# <div align="center">Noah's Data Science Portfolio</div>

<br>

## 1. Predicting House Sale Price

Skills Demonstrated: *multi-variate analysis, machine learning, linear regression, data transformation, exploratory data analysis*<br/>
Libraries and Programs: *Python, Jupyter Notebook, matplotlib, numpy, pandas, scipy, seaborn, sklearn*<br/>

### <div align="center">Project Overview</div>
Using a competition dataset available on Kaggle<sup>1</sup>, I created a model that can predict the sale price of a house in Ames, Iowa from 79 explanatory variables (features). The purpose of this project was to demonstrate my ability to tackle a complex dataset and model the behavior of a target variable. For a more in-depth look at this analysis, please refer to my [Jupyter Notebook](https://github.com/nphorsley59/House_Prices/blob/master/House_Prices.ipynb).

### <div align="center">Data Preparation</div>

#### 1. Exploration
I began by familiarizing myself with the Ames, Iowa housing dataset. It was divided into a [train](https://github.com/nphorsley59/House_Prices/blob/master/train.csv) and [test](https://github.com/nphorsley59/House_Prices/blob/master/test.csv) sample, each consisting of roughly 1,500 entries. Each entry held 78 features characterizing the house, lot, and surrounding area. It was the explicit goal of the competition to use the "train" sample (where 'Sale Price' was provided) to predict the 'Sale Price' of houses in the "test" sample.<br />

Due to the complexity of the dataset (Figure 1), a [detailed description](https://github.com/nphorsley59/House_Prices/blob/master/Data_Description.txt) of each of the 78 features was included. I reviewed this information and broke down the full feature list by type (numerical vs categorical) and role (building i.e. describes physical characteristics of house, space i.e. describes size of house, and location i.e. describes the surrounding area) in an [Excel spreadsheet](https://github.com/nphorsley59/House_Prices/blob/master/Feature_Log.xlsx). I also made predictions about the influence of each feature and 'Sale Price' and kept notes throughout the analysis process.<br />

**Figure 1.** Shape of training dataset, demonstrating the complexity of the dataset.<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/Train_Shape.png "Raw Train Dataset")

#### 2. Cleaning
I addressed several major issues during the cleaning process. First, missing values were widespread in both samples (Figure 2). I assigned a value of 'None' or '0' when 'NaN' clearly represented an entry that lacked the described feature (i.e. 'GarageType' for a house that doesn't have a garage). I dealt with other missing values on a feature-by-feature basis, using whichever method was appropriate. Second, I checked each entry for inconsistencies among shared features (i.e. 'GarageYrBlt', 'GarageFinish', 'GarageQual', etc. all describe a garage). Finally, I checked for typos and dropped uninformative features. Most of the cleaning required for this dataset was fairly lightweight, especially in the "train" sample.<br />

NOTE: The full dataset remained separated into "train" and "test" samples for cleaning to avoid [data leakage](https://machinelearningmastery.com/data-leakage-machine-learning/).<br />

**Figure 2.** Summary of missing data in the "test" sample.<br />

![alt_text](https://github.com/nphorsley59/House_Prices/blob/master/Figures/MissingData.png "Missing Data")

#### 3. Feature Engineering
The purpose of this step was to simplify the dataset, create new features that could inform the model, and ensure the structure of each feature was conducive to analysis. I began by merging the "train" and "test" samples to ensure changes were reflected in both. I then removed several uninformative features, including 'Id', 'Utilities', and 'PoolQC', and changed the data type for several others. I only wanted to keep features that could influence Sale Price, were known for most of the dataset, and contained variation. I also wanted to ensure the data type reflected the substance of the feature. The final step was to encode the heirarchical features that were not already numeric (Figure 3). For example, 'BsmtQual' has a clear linear relationship with 'SalePrice'; higher quality basements are worth more money. Encoding allows the model to easily incorporate these features.

**Figure 3.** Correcting data types and heirarchical encoding of non-numeric features.<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/Heir_Encoding.png "Feature Engineering")

### <div align="center">Quantitative Features</div>
Now that the dataset had been cleaned and organized, it was time for an exploratory data analysis. This is especially valuable in such a large, complex dataset, where a few features may hold most of the predictive power for 'SalePrice'. I started by making a correlation heatmap of the quantitative features (Figure 4). 'OverallQual', 'GrLivArea', and 'ExterQual' came out as the strongest quantitative predictors of 'SalePrice'. These features will undoubtedly play a significant role in my regression model. I also noted strong relationships between 'TotRmsAbvGrd' and 'GrLivArea', 'FullBath' and 'TotBath', '1stFlrSF' and 'TotalBsmtSF', and 'GarageArea' and 'GarageCars'. In each of these cases, I should consider removing the weaker predictor from the dataset.<br/>

**Figure 4.** A correlation heatmap of all quantitative features, showing meaningful predictive features and potential multicollinearity.<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/CorrMap_15.jpg "Correlation Heatmap")<br/>

Next, I explored my strongest quantitative predictors independently and removed any obvious outliers (Figure 5). Several of these features had heteroscedastic distributions which would need to be normalized in a future step.  

**Figure 5.** Scatterplot of 'GrLivArea' and 'SalePrice'; outliers circled in black.<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/LivingArea_Scatter1.png "Outliers")

### <div align="center">Qualitative Features</div>

I continued my exploratory data analysis by visualizing some of the qualitative features I expected to influence 'SalePrice'. Several of these described the location of the house and could be related, so I visualized them independently and together (Figure 6, 7). In general, I expected most qualitative features to have a weak influence on 'SalePrice' and found very few strong relationships.<br/>

**Figure 6.** Box and whisker plot showing variation in sale price for four qualitative variables. 'MSSubClass' describes the housing type and 'MSZoning' describes the zoning of the surrounding area.<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/Qual_Feat_Boxplots.jpg "Qualitative Features")<br/>

**Figure 7.** Swarmplot showing relationship between 'Neighborhood', 'MSZoning', and 'SalePrice'.<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/NeighbrhdZoning.jpg "Qualitative Relationships")

### <div align="center">Normalize Data</div>

#### 1. Response Variable
Before modeling, I checked the distribution and normality of my data. This was especially important for the response (target) variable, so that's where I began. I found that 'SalePrice' was skewed left quite significantly (Figure 8). To fix this, I performed a log(x+1) transformation (Figure 9).<br/>

**Figure 8.** The raw distribution (blue curve) of 'SalePrice' compared with a normal distribution (black curve).<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/Raw_Distribution.png "Raw Distribution")<br/>

**Figure 9.** The transformed distribution (blue curve) of 'SalePrice' compared with a normal distribution (black curve).<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/Log_Distribution.png "Log-Transformed Distribution")<br/>

#### 2. Explanatory Variables
I was also interested in tranforming particularly skewed explanatory variables (features). I set a cutoff of skew >= 1 and used a Box Cox transformation. I visually inspected my strongest predictors from my exploratory data analysis and was satisfied with the results (Figure 10). Notice that the heteroscedasticity noted earlier has been corrected.<br/>

**Figure 10.** Scatterplot of 'GrLivArea' and 'SalePrice' after a Box Cox transformation.<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/LivingArea_Scatter3.jpg)

### <div align="center">Regression Modeling</div>

#### 1. Preparation
A few final steps were required to prepare the dataset for training and testing models. First, I turned all qualitative features into [dummy variables](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)). Then, I separated the "test" data from the "train" data; 'SalePrice' is unknown for the "test" data, so it won't be helpful for model building. Finally, I further separated the "train" data into a "train" group and a "test" group. The "train" group was used to inform the model and the "test" group was used to test the model's accuracy. I did this step manually to create visualizations, but when actually testing models this was replaced with automated [cross-validation](https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f) executed by a custom function.

#### 2. Building and Testing Models
I chose to mostly use regularized linear models due to the complexity of the dataset<sup>2</sup>. These types of models help reduce overfitting. I also preprocessed the data using [RobustScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) to reduce the influence of outliers that weren't manually removed. This resulted in a Ridge Regression model, Lasso Regression model, and an Elastic Net Regression model, which each use different techniques for constraining the weights of the parameters (Figure 11). In addition to regularized linear models, I used a Gradient Boost algorithm, which essentially compares predictors to their predecessors and learns from the errors. Each of these models performed fairly well, producing RMSE (root mean square error) values around 0.10-0.15.<br/>

**Figure 11.** Visual comparison of the performance of four unique models built using machine learning algorithms.<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/Regression_Models.jpg "Regression Models")<br/>

To test these models more rigorously, I used a cross-validation function inspired by Serigne's notebook<sup>3</sup> on Kaggle (Figure 12).

#### 3. Stacking Models
Stacking is an [ensemble method](https://towardsdatascience.com/ensemble-methods-in-machine-learning-what-are-they-and-why-use-them-68ec3f9fef5f#:~:text=Ensemble%20methods%20is%20a%20machine,machine%20learning%20and%20model%20building.) that can improve the accuracy of model predictions by combining the strenghts of multiple models. This is an advanced method that I am still in the process of learning (it is not supported by scikit-learn) and implementing. I am also not sure it is entirely necessary for this dataset.

### <div align="center">Submission</div>
Now that I had several relatively accurate models, I used the model with the best cross-validation score to predict 'SalePrice' for the "test" sample. I included the [results](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/submission.csv) as a csv in this repository. Unfortunately, because this dataset was a competition dataset, I could not directly test the final accuracy of my model. I did find that the distribution of my 'SalePrice' predictions closely matched the distribution of 'SalePrice' for the training dataset (Figure 12). I considered this to be support for the strength of my model.<br/>

**Figure 12.** Comparison of 'SalePrice' distribution between the "train" and predicted "test" samples.<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/submission_comparison.jpg "Submission Comparison")

### <div align="center">Resources</div>
<sup>1</sup> https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview<br/>
<sup>2</sup> https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/<br/>
<sup>3</sup> https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard<br/>

<br>

## 2. Modeling Population Growth
Skills Demonstrated: *data simulation, predictive modeling, custom functions, feature engineering*<br />
Libraries and Programs: *Python, Jupyter Notebook, math, matplotlib, numpy, pandas*<br />

### <div align="center">Project Overview</div>
As part of my Master's thesis, I built a partial life-cycle matrix model<sup>1</sup> and used it to predict the survival rate of Common Grackles<sup>2</sup> during the non-breeding season. The model simulates realistic population growth and has several additional features that extend beyond the scope of my research. I am in the process of publishing the results produced by this model. I am also using this model to do predictive modeling for another publication investigating Micronesian Starling demography on the island of Guam. For a more in-depth look at this analysis, please refer to my [Jupyter Notebook](https://github.com/nphorsley59/Population_Growth_Modeling/blob/master/PLC_MatrixModel.ipynb).

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

<br>

## 3. Monitoring Avian Diversity
Skills Demonstrated: *data sourcing, data wrangling, data cleaning, lambda functions, data visualization*<br />
Libraries and Programs: *Python, Spyder, Tableau, numpy, pandas*<br />

### <div align="center">Project Overview</div>
In 2019, a colleague and I launched the Monitoring of Beneficial Birds in Agricultural Ecosystems Initiative. The purpose of this project is to connect sustainable land use practices with changes in native bird communities. As the sole analyst, I am responsible for transforming raw data into a format that can be quickly and easily communicated with landowners and funding agencies. For the Spring 2020 dataset, I used standard data wrangling, data cleaning, and data visualization techniques to create an interactive report. For a more in-depth look at this analysis, please refer to my Python scripts ([Create Dict](https://github.com/nphorsley59/Monitoring_Avian_Diversity/blob/master/Create_Dict.py), [Data Cleaning](https://github.com/nphorsley59/Monitoring_Avian_Diversity/blob/master/Data_Cleaning.py)).

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
The second phase was to complete tasks identified in Phase 1. I used common indexing functions, such as .loc/iloc and .at/iat, to identify and address typos and other minor errors. More widespread problems were addressed using more powerful functions and techniques, such as .replace(), .fillna(), lambda functions, loops, and custom functions (Figure 5).

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

I used Tableau to visualize the results of our Spring 2020 surveys. I have included a sample plot below (Figure 8). The full workbook<sup>3</sup> can be found on Tableau Public.<br />

**Figure 8.** A bar chart showing prominent members of the bird community (>10 individuals) at each site.

![alt text](https://github.com/nphorsley59/Portfolio/blob/master/AAD_Figures/BirdCbS_Sp2020_Table1.png "Bird Community by Site")<br />

### <div align="center">Resources</div>
<sup>1</sup> https://www.cowcreekorganics.com/about<br />
<sup>2</sup> https://www.birdpop.org/docs/misc/Alpha_codes_eng.pdf<br />
<sup>3</sup> https://public.tableau.com/profile/noah.horsley#!/<br />
