Link to Dataset: https://datahack.analyticsvidhya.com/contest/janatahack-healthcare-analytics-ii/#ProblemStatement
# Problem Statement

Recent Covid-19 Pandemic has raised alarms over one of the most overlooked area to focus: Healthcare Management. While healthcare management has various 
use cases for using data science, patient length of stay is one critical parameter to observe and predict if one wants to improve the efficiency of the 
healthcare management in a hospital. 

This parameter helps hospitals to identify patients of high LOS risk (patients who will stay longer) at the time of admission. 
Once identified, patients with high LOS risk can have their treatment plan optimized to miminize LOS and lower the chance of staff/visitor infection. 
Also, prior knowledge of LOS can aid in logistics such as room and bed allocation planning.

Suppose you have been hired as Data Scientist of HealthMan – a not for profit organization dedicated to manage the functioning of Hospitals 
in a professional and optimal manner.The task is to accurately predict the Length of Stay for each patient on case by case basis so that the 
Hospitals can use this information for optimal resource allocation and better functioning. The length of stay is divided into 11 different classes 
ranging from 0-10 days to more than 100 days.

### Evaluation Metric
100 * Accuracy

### Methods
The dataset contains over 380,000 rows and 18 features.
The data was first trained using all the original features to get the baseline scores. After these baseline models, the best performing models were used in the final models,
which were Lightgbm and Catboost models.

Baseline Models (LB Scores)

                               Final Score            Private Scores
      Cat bline	              42.7000109445113	      42.5329847388581
      LGB bline	              42.4592316953048	      42.4040858515231
      Gbm bline	              42.2056838495495	      42.0490058977321
      extrees bline	          41.2717522162635	      40.8925639934334
      Logreg bline	          40.5202291051038	      39.9927038365659


Final Models (LB Scores)

                                    Final Score                   Private Scores
      blend (lgb and cat)	          43.1396154828354	            42.8990089378002
      lgb1	                        42.8386414213272	            42.4746154313857
      cat1	                        43.0119295173471	            42.9427859184046

### Note:
Predicting the length of stay of a patient is very difficult to estimate even for a human, hence the low accuracy is expected. 
There are many factors that could affect the length of stay of a patient. Such as the severity of illness, the patient admitted, their medical history etc.
