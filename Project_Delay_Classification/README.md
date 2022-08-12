## Background
%X of construction projects are delayed. In addition, the projects that meet the deadline also experience delays, however, mitigated before impacting the project schedule. Therefore, early identification of potential project delay will help mitigate the project level impact on sechdule. This paper proposes a machine learning approach to classify projects as high-risk/low risk based on the combination of reasons for milestone-delays.

## Methodology
Decision tree was used to classify the projects into "high-risk", and "low-risk" based on the occurance of different types of delay, during the entire project. 

## Data Colleciton
The dataset was provided by the company.The initial data consisted of 265 rows and 13 columns. Each row corresponds to a unique project, consisting of number of milestones. During the weekly meetings, the expected end dates for the milestones were compared with the actucal progress of work. If the project milestone experinced a delay, the reason for the delay was stored and the milestone schedule was updated to a new date. This process was repeated for all 265 projects. 

## Data cleaning
During the data cleaning process, the irrelevant columns were dropped. The resulting set consisted of 256 rows and two columns, corresponding to the project ID and the "reason for miss" column. The "reason for miss" column consists of 28 categories of causes for delay, as shown in figure X. The 28 categories were renamed to "F1" to "F28". The amount of delay in the target column was encoded into two classes of 0 (low risk) and 1 (high risk) based on the value less/greater the median of the column (9 days). Finally the dataset was transformed in Power BI resulting in large table with 265 rows and 28 columns for features, and one column for the target. The values in each row corresponded to the number of occurance of each "reason for miss" for each project. The target value for each project indicates the sum of the number of reasons for miss.XXXXXXXXXXX Change this to number of days missed at project level, XXXXXXXXXXX


## Data Preprocessing
https://monkeylearn.com/blog/data-preprocessing/
B
Data quality assurance, data cleaning, data transformaiton, and data reduction was performed. 

## Description of the Dataset
The dataset inlcudes scheduling data for 265 projects. The data was devided into training/testing set by %70/%30 ratio.

## Results and Discussion
Performance evaluation of the proposed method in the test dataset resulted %X precision, %X recall, F1-score of X, as visually depicted below. 




## Corelation Matrix for Delay Root Causes

Figure below shows the corelation between the different features of the project, based on 256 projects. Here features F0 to F26 indicate the identified root causes of delay, while the saturated colors indicate higher levels are positive corelation between the features. For example there is a relatively significant positive corelation between F2 ( Change orders) and F4 (Supply chain).

![corelation_matrix](https://user-images.githubusercontent.com/55706949/183313326-4162c311-c947-45dd-9840-06b1e6535e8d.png)

