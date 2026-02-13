    Project workflow for taxi_classification_project

1) Data Engineering
The raw data was spread across multiple sources—driver profiles, ride records, vehicle details, and incident reports. I merged these datasets into a single, cohesive table by aggregating key metrics for each driver. This included calculating totals like the number of complaints, incidents, and rejected rides, as well as derived features such as success rates and time since last vehicle inspection. 

=> The result was a clean, structured dataset ready for modeling, saved as collected.csv

2) Train/Test Split
To ensure reliable evaluation, I split the data into training and test sets using a 70/30 ratio. I used stratified sampling to maintain the original class distribution in both sets, which helped prevent bias in the model’s performance metrics

3) Preprocessing
Before modelling, I addressed common data issues:
    - Missing values : for example, missing driver ages were imputed using the mean age from the train dataset only (to avoid dataleakge !!!) 
    - Categorical values : I applied one-hot encoding to variables like car models, since there’s no inherent order between categories (e.g., a Toyota isn’t "better" than a Honda)
    - Numerical variables : Features like tip amounts were standardized
    - Target variable : The class labels (A or B) were encoded as 0 and 1 for compatibility with scikit-learn

4) Modelling
I started with a simple baseline model—logistic regression—to establish a performance benchmark. Then, I trained a more complex Random Forest model, which better captured non-linear relationships in the data
My goal is to maximise recall while keeping precision relatively high (0.5)

5) Result: 
Validation recall:  0.9333333333333333
F1_score final:  0.8615384615384616
Confusion matrix: 
[[113   7]
 [  2  28]]