## Background
%X of construction projects are delayed. In addition, the projects that meet the deadline also experience delays, however, mitigated before impacting the project schedule. Therefore, early identification of potential project delay will help mitigate the project level impact on sechdule. This paper proposes a machine learning approach to classify projects as high-risk/low risk based on the combination of reasons for milestone-delays.

## Methodology
Decision tree was used to classify the projects into "high-risk", and "low-risk" based on the occurance of different causes for delay, during the entire project. 

## Data Colleciton
Raw schdule data was provided by the company. 

## Data Preprocessing
https://monkeylearn.com/blog/data-preprocessing/
Data quality assurance, data cleaning, data transformaiton, and data reduction was performed. 

## Description of the Dataset
The dataset inlcudes scheduling data for 265 projects. The data was devided into training/testing set by %70/%30 ratio.

## Results and Discussion
Performance evaluation of the proposed method in the test dataset resulted %X precision, %X recall, F1-score of X, as visually depicted below. 




## Corelation Matrix for Delay Root Causes

Figure below shows the corelation between the different features of the project, based on 256 projects. Here features F0 to F26 indicate the identified root causes of delay, while the saturated colors indicate higher levels are positive corelation between the features. For example there is a relatively significant positive corelation between F2 ( Change orders) and F4 (Supply chain).

![corelation_matrix](https://user-images.githubusercontent.com/55706949/183313326-4162c311-c947-45dd-9840-06b1e6535e8d.png)

