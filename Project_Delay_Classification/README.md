## Background
Majority of construction projects are delayed. In addition, the projects that meet the deadline also experience delays, however, mitigated before impacting the project schedule. Therefore, early identification of potential project delay will help mitigate the project level impact on sechdule. This paper proposes a machine learning approach to classify projects as high-risk/low risk based on the combination of reasons for milestone-delays.

## Methodology
Decision tree was used to classify the projects into "high-risk", and "low-risk" based on the occurance of different types of delay, during the entire project. 

## Data Colleciton
The dataset was provided by the company. The initial data consisted of 265 rows and 13 columns. Each row corresponds to a unique project, consisting of number of milestones. The records were compiled during the  the weekly meetings, where the expected end dates for the milestones were compared with the actucal progress of work. If the project milestone experinced a delay, the reason for the delay was entered and the milestone schedule was updated to a new date. This process was repeated for all 265 projects. 

## Data cleaning
During the data cleaning process, the irrelevant columns from the initial dataset were dropped. The resulting set consisted of 256 rows and three columns, corresponding to the project ID, the "reason for miss", and the amount of delay for each "reason for miss" column. The delays assigned to "CC sign off" milestone were treated as the project level delays, therefore, to fasciliate data wranglining the dataset was devided into two tables with the schema of (Scope, Reason for delay, Amount of Delay) and (Scope, Sign off delay (label), Amount of Delay). Next, the frequency of delays, and sum of sign-off delays pers scope were calculated and the tables were joined using th left inner join. The projects with null sign-off value were discarded, which resulted in 144 projects after join. Next, the delay amount and the frequency of occurnace for each reason-for-miss were normalized to values in range of [0 1]. 

 and stored in a seperate table.  The "reason for miss" column consists of 28 categories of causes for delay, as shown in figure X. The 28 categories were renamed to "F0" to "F25" holding the frequency of occurance, and the amount of delay for each type of delay for each project.

The amount of delay in the target column was encoded into two classes of 0 (low risk) and 1 (high risk) based on the value less/greater the median of the column. Finally the dataset was transformed in Power BI resulting in a large table with 265 rows and 26*2 +1 columns for features, and the target. The values in each row corresponded to the number of occurance of each "reason for miss" for each project. The target value for each project indicates the sum of the number of reasons for miss.XXXXXXXXXXX Change this to number of days missed at project level, XXXXXXXXXXX


## Data Preprocessing
https://monkeylearn.com/blog/data-preprocessing/
B
Data quality assurance, data cleaning, data transformaiton, and data reduction was performed. 

## Description of the Dataset
The dataset inlcudes scheduling data for 265 projects. The data was devided into training/testing set by %70/%30 ratio.

## Results and Discussion
Performance evaluation of the proposed method in the test dataset resulted %X precision, %X recall, F1-score of X, as visually depicted below. 




## Corelation Matrix for Delay Root Causes

Figure below shows the corelation between the different features of the project, based on 256 projects. Here features F0 to F26 indicate the identified root causes of delay, while the saturated colors indicate higher levels are positive corelation between the features. For example there is a relatively significant positive corelation between ( Change orders) and (Supply chain).

![corelation_matrix](https://user-images.githubusercontent.com/55706949/183313326-4162c311-c947-45dd-9840-06b1e6535e8d.png)

