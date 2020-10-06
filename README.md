# <div align="center">Noah's Data Science Portfolio</div>

## Table of Contents
* [Predicting Passenger Survival - Classification](https://github.com/nphorsley59/PORTFOLIO/blob/master/README.md#1-predicting-passenger-survival)
* [Predicting Sale Price - Regression](https://github.com/nphorsley59/PORTFOLIO#1-predicting-house-sale-price)
* [Modeling Population Growth - Simulation](https://github.com/nphorsley59/PORTFOLIO#2-modeling-population-growth)
* [MNIST Digit Recognition - Clustering](https://github.com/nphorsley59/PORTFOLIO#3-mnist-digit-recognition) 

<br/>

## 1. Predicting Passenger Survival

Skills Demonstrated: *classification, ensemble learning, hyperparameter tuning, EDA, feature engineering*<br/>
Libraries and Programs: *Python, Jupyter Notebook, pandas, pivot_table, matplotlib, numpy, regex, sklearn, seaborn*<br/>

### <div align="center">Project Overview</div>
Using the Titanic competition dataset available on Kaggle<sup>1</sup>, I created models to predict which passengers would survive the 1912 sinking of the Titanic. The dataset included passenger attributes such as Name, Age, Sex, and Class, as well as information about their trip, such as their Cabin, Embarkment Location, and Ticket Price. My top model scored 83% accuracy in cross-validation using a Soft Voting Classifier, which which would put me in the top 5% of Kaggle submissions. For a more in-depth look at this analysis, please refer to my [Jupyter Notebook](https://github.com/nphorsley59/Passenger_Survival/blob/master/Predicting_Passenger_Survival.ipynb).<br>

## <div align="center">Exploratory Data Analysis</div>
### 1. Explore Data Structure
I began by learning some basics about the dataset (Figure 1). I wanted to know its shape (shape()), its column names (columns()), and its contents (sample(), describe(), info()).<br>

**Figure 1.** A sample of ten rows from the training set.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/data_sample.png "Data Sample")<br>

Next, I inspected the distribution of each feature<sup>3</sup>. For numerical features, I calculated location (mean(), trimmed_mean(), median()) and variation (std(), mad()), and for categorical features, I inspected distribution using counts and proportions (value_counts()).<br>

**Figure 2.** A kernel density estimation of the distribution of passenger fare and a pie chart of the distribution of passenger class.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/dist_classandfare.jpg "Feature Distributions")<br>

### 2. Clean and Organize
I identified several issues while exploring the dataset. I prefer to clean these up before exploring relationships between variables.<br>

For this dataset, I needed to:<br>
- Address NaNs<br>
- Split 'Cabin' into deck and room number<br>
- Split 'Name' into title and last name<br>
- Use 'Ticket', 'ParCh', and 'SibSp' to determine if passengers were traveling alone or in a group<br>
- Apply log(x+1) transformation to 'Fare' to fix right-skew<br>
- Streamline the dataset(drop/rename columns, change dtypes, etc)<br>

#### 2.1. Complete Columns with NaNs
I mapped NaNs and completed columns that had relatively straight-forward solutions. I assigned a placeholder value for NaNs in 'Cabin' and 'Age' until I could address them properly.<br>

**Figure 3.** Tables showing NaNs by feature for the 'train' and 'test' datasets.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/null_tables.png "NULL value Tables")<br>

#### 2.2. Split 'Cabin'
**Figure 4.** I used the string matching libarary, re, to parse deck and cabin number from 'Cabin'.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/cabin_split.png "Splitting 'Cabin'")<br>

#### 2.3. Split 'Name'
**Figure 5.** I parsed 'Title' and 'Last' from 'Name' and reduced low frequency 'Title' results ("Col", "Jonkheer", "Rev") to "Other".<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/name_split.png "Splitting 'Name'")<br>

#### 2.4. Engineer 'GroupSize' and 'FamilySize'
**Figure 6.** I counted ticket replicates to identify non-familial groups and summed 'ParCh' to 'SibSp' to identify familial groups.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/partysize_split.png "Engineering 'Connections'")<br>

### 3. Examine Relationships
I concluded by exploring how features were related to the target, 'Survived', and to each other. Before looking at individual features, I constructed a correlation matrix and visualized it as a heatmap.<br>

**Figure 7.** Correlation coefficients for linear relationships between features.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/corr_heatmap2.png "Correlation Heatmap")<br>

#### 3.1. Collinearity
Several features were strongly correlated, introducing collinearity into the model. I explored them further to determine which were appropriate to keep, drop, or engineer for analysis.<br>

**Figure 8.** A swarm plot of deck and cabin assignments as well as the fate of their occupants.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/cabin_deck2.png "Deck and Cabin Assignments")<br>

A passenger's cabin assignment had little impact on their fate. Considering 'Cabin' and 'Deck' were unknown for ~80% of passengers in the dataset, I decided to drop these features from the analysis.<br>

**Figure 9.** Survival rate based on various criteria describing a passenger's connections on-board.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/partysize_plot.png "Party Info Plot")<br>

I found that being alone or being in a group of more than four seemed to decrease a passenger's chance of surviving. I engineered a new feature, 'Connections', and binned it based on these findings (group size of 1, 2-4, and >4).

#### 3.2. Complete 'Age'
Passenger age was unknown for ~20% of the dataset. I grouped passengers with known age by 'Sex', 'Title', and 'Class' - features correlated with 'Age' - and calculated the median age for each combination. Then, to complete 'Age', I filled all passenger records of unknown age with the appropriate group median (matching 'Sex', 'Title' and 'Class'). I got this idea from a Kaggle notebook by Manav Sehgal<sup>4</sup>.

**Figure 10.** The loop used to complete 'Age'.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/age_code.png "Completing 'Age'")<br>

#### 3.3. Address 'Fare' Distribution
While examining 'Fare' and how it related to other features, I noted two problems:<br>
- A handful of passengers had a ticket fare of $0.00<br>
- Passengers who shared tickets paid more than the average fare for their class<br>

**Figure 11.** Fare was positively related to 'Connections', even when controlling for 'Class'.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/fare_connections_unadj.png "Influence of Connections on Fare")<br>

From this, I concluded that the fare for shared tickets must be a lump sum rather than an individual fare. I addressed this by dividing 'Fare' by 'GroupSize'. I also concluded that passengers with a ticket fare of $0.00 were crew members. They were all middle-aged males and almost all of them died. I addressed this by assigning them a new 'Class'.<br>

#### 3.4. Multivariate Relationships
I looked at many multivariate relationships and included figures for two that I found particularly interesting/informative. The age and sex of a passenger were strong predictors of survival; however, on top of that, class was arguably even more important. Age and sex can be condensed into title to show the relationship with class. I found this complex web of influence very interesting.

**Figure 12.** Violin plot showing how the age and sex of a passenger influenced their chance of survival.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/age_sex.png "Age, Sex, Survival")<br>

**Figure 13.** The influence of title - a rough estimate of age and sex - and class on passenger survival.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/title_class.png "Title, Class, Survival")<br>

I prepared the dataset for modeling by dropping uninformative columns, encoding all non-integer/non-float data types, splitting 'train' into a training and testing set for cross-validation, and scaling the data.<br>

## <div align="center">Modeling</div>
My goal was to build a model that could accurately predict the fate of a passenger on the Titanic. Considering the simplicity of the dataset and the unquantifiable forces at play in the real event, I set a goal of 80% model accuracy.<br>

### 1. Pre-processing
I prepared the dataset for modeling by dropping uninformative columns, encoding all non-integer/non-float dtypes, splitting 'train' into a training and testing set for cross-validation, and scaling the data.<br>

**Figure 14.** A sample from the streamlined 'train' set, pre-scaling.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/final_dataset.png "'Train' prepped for modeling")<br>

**Figure 15.** The code used to split 'train' and scale the data for modeling.<br>

![alt_text](https://github.com/nphorsley59/Passenger_Survival/blob/master/Figures/train_test_scale.png "Split and Scale")<br>

### 2. Classification
I tested a range of classification algorithms, including Logistic Regression, Support Vector Machine, K-nearest Neighbors, and Decision Tree<sup>5</sup>. I used the training set to fit the model and the testing set to predict survival and score model accuracy (classification_report()). I also used GridSearchCV() to tune the hyperparameters for each model.<br>

**Figure 16.** Code used to fit and score a Logistic Regression model.<br>

![alt_text](https://github.com/nphorsley59/Predicting_Passenger_Survival/blob/master/Figures/log_reg_code.png "Logistic Regression Code")<br>

**Figure 17.** A grid search with cross-validation to determine the best hyperparameters for the K-nearest Neighbors model.<br>

![alt_text](https://github.com/nphorsley59/Passenger_Survival/blob/master/Figures/grid_search.png "Tuning Hyperparameters")<br>

### 3. Ensemble Learning
I used ensemble learning to construct models with the aggregate knowledge of many simpler models. Specifically, I used Random Forest (many Decision Trees) and a Voting Classifier (a mix of classification algorithms). I then scored and compared my models based on precision, recall, and accuracy.<br>

**Figure 18.** Feature importances generated by the Random Forest model.<br>

![alt_text](https://github.com/nphorsley59/Passenger_Survival/blob/master/Figures/feature_importance.png "Feature Importance")<br>

**Figure 19.** The decision Boundary from the Voting Classifier for 'Fare' and 'Age'. Only passengers that died are shown.<br>

![alt_text](https://github.com/nphorsley59/Passenger_Survival/blob/master/Figures/decision_boundary_death.png "Decision Boundary")<br>

**Figure 20.** A bar chart showing model accuracy, one of several scores used to determine model selection.<br>

![alt_text](https://github.com/nphorsley59/Passenger_Survival/blob/master/Figures/model_selection.png "Model Selection")<br>

### 4. Final Predictions
The most balanced, high-scoring model used a Voting Classifier to predict passenger survival to 83% accuracy. It was better at predicting death than survival and struggled the most with false negatives. This model was used to predict survival in the 'test' set for submission.<br>

**Figure 21.** A normalized confusion matrix for the Voting Classifier model.<br>

![alt_text](https://github.com/nphorsley59/Passenger_Survival/blob/master/Figures/conf_matrix.png "Confusion Matrix")

## <div align="center">Concluding Thoughts</div>
This project was my first full-length analysis using classification algorithms and ensemble learning. I learned some new tools for visualizing these models and had a lot of fun seeing how different hyperparameters effected the predictions. I'm sure there is room for incremental improvement in this analysis, in both feature engineering and modeling, but I was generally quite satisfied with my results. Most other Kaggle submissions that used cross-verification (controls for overfitting) predicted survival to 75-80% accuracy. Moving forward, I'd like to work with other ensemble learning techniques, such as Gradient Boosting and stacking, and improve my EDA process to include a principle components analysis. Thanks for reading!<br>

## <div align="center">Resources</div>
<sup>1</sup> https://www.kaggle.com/c/titanic <br/>
<sup>2</sup> https://www.encyclopedia-titanica.org/cabins.html <br/>
<sup>3</sup> https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/ <br>
<sup>4</sup> https://www.kaggle.com/startupsci <br>
<sup>5</sup> https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/<br>

<br>

## 2. Predicting House Sale Price

Skills Demonstrated: *machine learning, linear regression, data transformation, exploratory data analysis*<br/>
Libraries and Programs: *Python, Jupyter Notebook, matplotlib, numpy, pandas, scipy, seaborn, sklearn*<br/>

### <div align="center">Project Overview</div>
Using a competition dataset available on Kaggle<sup>1</sup>, I created a model that can predict the sale price of a house in Ames, Iowa from 79 explanatory variables (features). The purpose of this project was to demonstrate my ability to tackle a complex dataset and model the behavior of a target variable. For a more in-depth look at this analysis, please refer to my [Jupyter Notebook](https://github.com/nphorsley59/House_Prices/blob/master/House_Prices.ipynb).

### <div align="center">Data Preparation</div>

### 1. Exploration
I began by familiarizing myself with the Ames, Iowa housing dataset. It was divided into a [train](https://github.com/nphorsley59/House_Prices/blob/master/train.csv) and [test](https://github.com/nphorsley59/House_Prices/blob/master/test.csv) sample, each consisting of roughly 1,500 entries. Each entry held 78 features characterizing the house, lot, and surrounding area. It was the explicit goal of the competition to use the "train" sample (where 'Sale Price' was provided) to predict the 'Sale Price' of houses in the "test" sample.<br />

Due to the complexity of the dataset (Figure 1), a [detailed description](https://github.com/nphorsley59/House_Prices/blob/master/Data_Description.txt) of each of the 78 features was included. I reviewed this information and broke down the full feature list by type (numerical vs categorical) and role (building i.e. describes physical characteristics of house, space i.e. describes size of house, and location i.e. describes the surrounding area) in an [Excel spreadsheet](https://github.com/nphorsley59/House_Prices/blob/master/Feature_Log.xlsx). I also made predictions about the influence of each feature and 'Sale Price' and kept notes throughout the analysis process.<br />

**Figure 1.** Shape of training dataset, demonstrating the complexity of the dataset.<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/Train_Shape.png "Raw Train Dataset")

### 2. Cleaning
I addressed several major issues during the cleaning process. First, missing values were widespread in both samples (Figure 2). I assigned a value of 'None' or '0' when 'NaN' clearly represented an entry that lacked the described feature (i.e. 'GarageType' for a house that doesn't have a garage). I dealt with other missing values on a feature-by-feature basis, using whichever method was appropriate. Second, I checked each entry for inconsistencies among shared features (i.e. 'GarageYrBlt', 'GarageFinish', 'GarageQual', etc. all describe a garage). Finally, I checked for typos and dropped uninformative features. Most of the cleaning required for this dataset was fairly lightweight, especially in the "train" sample.<br />

NOTE: The full dataset remained separated into "train" and "test" samples for cleaning to avoid [data leakage](https://machinelearningmastery.com/data-leakage-machine-learning/).<br />

**Figure 2.** Summary of missing data in the "test" sample.<br />

![alt_text](https://github.com/nphorsley59/House_Prices/blob/master/Figures/MissingData.png "Missing Data")

### 3. Feature Engineering
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

### 1. Response Variable
Before modeling, I checked the distribution and normality of my data. This was especially important for the response (target) variable, so that's where I began. I found that 'SalePrice' was skewed left quite significantly (Figure 8). To fix this, I performed a log(x+1) transformation (Figure 9).<br/>

**Figure 8.** The raw distribution (blue curve) of 'SalePrice' compared with a normal distribution (black curve).<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/Raw_Distribution.png "Raw Distribution")<br/>

**Figure 9.** The transformed distribution (blue curve) of 'SalePrice' compared with a normal distribution (black curve).<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/Log_Distribution.png "Log-Transformed Distribution")<br/>

### 2. Explanatory Variables
I was also interested in tranforming particularly skewed explanatory variables (features). I set a cutoff of skew >= 1 and used a Box Cox transformation. I visually inspected my strongest predictors from my exploratory data analysis and was satisfied with the results (Figure 10). Notice that the heteroscedasticity noted earlier has been corrected.<br/>

**Figure 10.** Scatterplot of 'GrLivArea' and 'SalePrice' after a Box Cox transformation.<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/LivingArea_Scatter3.jpg)

### <div align="center">Regression Modeling</div>

### 1. Preparation
A few final steps were required to prepare the dataset for training and testing models. First, I turned all qualitative features into [dummy variables](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)). Then, I separated the "test" data from the "train" data; 'SalePrice' is unknown for the "test" data, so it won't be helpful for model building. Finally, I further separated the "train" data into a "train" group and a "test" group. The "train" group was used to inform the model and the "test" group was used to test the model's accuracy. I did this step manually to create visualizations, but when actually testing models this was replaced with automated [cross-validation](https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f) executed by a custom function.

### 2. Building and Testing Models
I chose to mostly use regularized linear models due to the complexity of the dataset<sup>2</sup>. These types of models help reduce overfitting. I also preprocessed the data using [RobustScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) to reduce the influence of outliers that weren't manually removed. This resulted in a Ridge Regression model, Lasso Regression model, and an Elastic Net Regression model, which each use different techniques for constraining the weights of the parameters (Figure 11). In addition to regularized linear models, I used a Gradient Boost algorithm, which essentially compares predictors to their predecessors and learns from the errors. Each of these models performed fairly well, producing RMSE (root mean square error) values around 0.10-0.15.<br/>

**Figure 11.** Visual comparison of the performance of four unique models built using machine learning algorithms.<br/>

![alt_text](https://github.com/nphorsley59/Predicting_Sale_Price/blob/master/Figures/Regression_Models.jpg "Regression Models")<br/>

To test these models more rigorously, I used a cross-validation function inspired by Serigne's notebook<sup>3</sup> on Kaggle (Figure 12).

### 3. Stacking Models
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

## 3. Modeling Population Growth
Skills Demonstrated: *data simulation, predictive modeling, custom functions, feature engineering*<br />
Libraries and Programs: *Python, Jupyter Notebook, math, matplotlib, numpy, pandas*<br />

### <div align="center">Project Overview</div>
As part of my Master's thesis, I built a partial life-cycle matrix model<sup>1</sup> and used it to predict the survival rate of Common Grackles<sup>2</sup> during the non-breeding season. The model simulates realistic population growth and has several additional features that extend beyond the scope of my research. I am in the process of publishing the results produced by this model. I am also using this model to do predictive modeling for another publication investigating Micronesian Starling demography on the island of Guam. For a more in-depth look at this analysis, please refer to my [Jupyter Notebook](https://github.com/nphorsley59/Population_Growth_Modeling/blob/master/PLC_MatrixModel.ipynb).

### <div align="center">Model Structure</div>

### Stage 1
The first modeling stage simulates population growth over a pre-defined trial period to transform a single population variable into an array representing stable age distribution. 

### Stage 2
The second modeling stage proportionally adjusts the stable age distribution array to represent the desired initial population size. From here, population growth is simulated over the length of time defined by the user.

### Stage 3
The third and final modeling stage averages annual population size and growth over the number of simulations defined by the user. It then plots mean population size with confidence intervals and calculates an annual trend estimate.

### Additional Features
The three stages outlined above describe the most simplistic modeling approach. More details can be found in the PLC_MatrixModel_AddFeat.txt file included in my Portfolio repository.

### <div align="center">Results</div>

Matrix models use matrices to track population growth across age classes (Table 1). This model structure allowed me to input age-specific survival and fecundity for more accurate modeling results. 

**Table 1.** A sample of the age matrix dataframe that shows population distribution across age classes.<br />

![alt text](https://github.com/nphorsley59/Population_Growth_Modeling/blob/master/Figures/Pop_Matrix_Table1.png "Age Matrix")<br />

In order to account for environmental stochasticity, I built natural variation into each rate based on real data collected in the field. As a result, each simulation produces slightly different results (Table 2, Figure 2). 

**Table 2.** A sample of the population growth dataframe that show population growth across simulations.<br />

![alt text](https://github.com/nphorsley59/Population_Growth_Modeling/blob/master/Figures/Pop_Growth_Table1.png "Population Growth")<br />

For my thesis, we used established demographic rates from the literature and estimated demographic rates from our own research (Table 3) to predict non-breeding season survival in three distinct populations: a stable population, the current global population, and the current Illinois population (Table 4, Figure 1). 

**Table 3.** The model parameters used to predict non-breeding season survival.<br />

![alt text](https://github.com/nphorsley59/Population_Growth_Modeling/blob/master/Figures/Model_Parameters_Table1.png "Model Parameters")<br />

**Table 4.** The predicted rates of non-breeding season survival for each population.<br />

![alt text](https://github.com/nphorsley59/Population_Growth_Modeling/blob/master/Figures/NBS_Survival_Predictions_Table1.png "Model Predictions")<br />

**Figure 1.** Population growth projections using predictions of non-breeding survival for three distinct populations.<br />
&nbsp;  

![alt text](https://github.com/nphorsley59/Population_Growth_Modeling/blob/master/Figures/Proj_Pop_Growth_Figure1.png "Predicted Population Growth")<br />

**Figure 2.** A sample of simulations produced by the model, replicating projected decline in Illinois.<br />
![alt text](https://github.com/nphorsley59/Population_Growth_Modeling/blob/master/Figures/livesim_plot_24sims.gif "Simulation Animation")

### <div align="center">Summary</div>

The partial life-cycle matrix model I built for my Master's thesis was used to project population decline in my study species, the Common Grackle, and to predict rates of non-breeding season survival for three distinct populations: a stable population, the current global population, and the current Illinois population. The modeling approach I chose allowed me to use many demographic parameters and account for realistic environmental variation. I learned a lot about model design, custom functions, and advanced visualization techniques from this project and am excited to reuse the model to answer other research questions in the future.

### <div align="center">Resources</div>
<sup>1</sup> https://onlinelibrary.wiley.com/doi/abs/10.1034/j.1600-0706.2001.930303.x<br />
<sup>2</sup> https://birdsoftheworld.org/bow/species/comgra/cur/introduction<br />

<br>

## 4. MNIST Digit Recognition
### <div align="center">Project Overview</div>
Skills Demonstrated: *SVM, KNN, big data, model optimization, data augmentation*<br />
Libraries and Programs: *Python, Jupyter Notebook, matplotlib, numpy, pandas, scikit-learn, scipy, statistics*<br />

Computer vision is a common application of machine learning algorithms. I used the MNIST dataset<sup>1</sup> to demonstrate the application of clustering methods<sup>2</sup> (SVM and KNN) to computer vision. The primary objective of the project was to train a model to recognize digital, black-and-white images of hand-written digits (0-9). My top model correctly identified >97% of the images in the 'test' dataset. For a more in-depth look at this analysis, please refer to my [Jupyter Notebook](https://github.com/nphorsley59/Handwritten_Digits/blob/master/digit_recognition_classifier.ipynb).

### <div align="center">Preparation</div>
Preparing the MNIST dataset for analysis was relatively straight-forward. Even though I assumed it was a clean dataset, I ran some quick tests looking for NaNs and other potential typos/outliers to be safe. I also collected some basic information about the structure of the dataset and visualized the images the model would be working with (Figure 1). The full dataset contained 42,000 28x28 images of digits, ranging from 0-9.

**Figure 1.** A sample of hand-written digits from the dataset.</br>

![alt_text](https://github.com/nphorsley59/MNIST_Digit_Recognition/blob/master/Figures/60_digits.png "Sample Digit")

### <div align="center">Modeling</div>

### 1. Support Vector Machine
I was interested in building several different models and comparing their performance. I started with a relatively simple clustering method, SVM. There were a few steps to this method:</br>
1) shuffle the 'train' rows</br>
2) split 'train' into Train and Test sets</br>
3) scale the Train set</br>
4) build and fit an SVM model</br>
5) test the accuracy of the model</br>
6) identify strengths and weaknesses of the method</br>

**Figure 2.** Fitting and scoring an SVM model and visualizing its confusion matrix.</br>

![alt_text](https://github.com/nphorsley59/Digit_Recognition/blob/master/Figures/SVM_1.png "SVM Model")</br>

The SVM model tested surprisingly well. Most digits were identified correctly (the numbers on the diagonal). However, the model did struggle to identify 8's and often misidentified digits as 2's. It also only had about 95% accuracy; good but not great.</br>

### 2. K-nearest Neighbors
K-nearest Neighbors (KNN) is another commonly used clustering algorithm. Similar to SVM, I split this analysis into several steps:</br>
1) shuffle the 'train' rows</br> 
2) split 'train' into Train and Test sets</br>
3) build and fit a KNN model</br>
4) use a grid search to hone the hyperparameters</br>
5) test the accuracy of the model</br>

The base KNN model slightly outperformed the SVM model I tested, but it could be improved. I performed a grid search to find better hyperparameter values.</br>

**Figure 3.** Executing a grid search to compare hyperparameters.</br>

![alt_text](https://github.com/nphorsley59/MNIST_Digit_Recognition/blob/master/Figures/GridSearch.png "KNN Grid Search")</br>

**Figure 4.** Fitting and scoring an adjusted KNN model.</br>

![alt_text](https://github.com/nphorsley59/MNIST_Digit_Recognition/blob/master/Figures/KNN_adj.png "Adjusted KNN Model")</br>

The adjusted KNN algorithm yielded an average cross-validation score of 97%, improving the image recognition accuracy of the model by 2%. For fun, I tried getting even closer to 100% accuracy with data augmentation.</br>

Data augmentation increases the effective sample size of a dataset without actually collecting more data. In this case, I shifted each digit up, down, left, and right, resulting in a training dataset that was 5x larger and had slightly more variation. This model took hours to run and compared almost 200,000 images, roughly 150 million data points. If I had more computing power, I could have added rotated images to the dataset as well.</br>

**Figure 5.** Original and shifted copies of a sample '2'.</br>

![alt_text](https://github.com/nphorsley59/MNIST_Digit_Recognition/blob/master/Figures/shifted_digits.png "Original and Shifted Digits")</br>

**Figure 6.** Model performance for cross-validation of the augmented 'train' dataset.</br>

![alt_text]()</br>

**Figure 7.** Top model performance on the 'test' dataset.</br>

![alt_text]()</br>

### <div align="center">Resources</div>
<sup>1</sup> https://www.kaggle.com/c/digit-recognizer<br/>
<sup>2</sup> https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/
