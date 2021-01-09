# Capstone Project: Azure Machine Learning Engineer

This is the Capstone project (last of the three projects) required for fulfillment of the Nanodegree Machine Learning Engineer with Microsoft Azure from Udacity. In this project, we use a dataset external to Azure ML ecosystem. 

Azure Machine Learning Service and Jupyter Notebook is used to train models using both Hyperdrive and Auto ML and then the best of these models is deployed as an HTTP REST endpoint. The model endpoint is also tested to verify if it is working as intented by sending an HTTP POST request. Azure ML Studio graphical interface is not used in the entire exercise to encourage use of code which is better suited for automation and gives a data scientist more control over their experiment.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
In this dataset, we predict divorce among couples by using the Divorce Predictors Scale (DPS) on the basis of [Gottman couples therapy](https://www.gottman.com/blog/an-introduction-to-the-gottman-method-of-relationship-therapy/). The data was collected from seven different regions of Turkey, predominantly from the Black Sea region. Of the participants, 84 (49%) were divorced and 86 (51%) were married couples. Divorced participants answered the scale items by considering their marriages whereas, of the married participants, only those with happy marriages, without any thought of divorce, were included in the study.

The dataset consists of 170 rows/records/examples and 54 features/attributes/columns. Attribute columns are labeled as `Atr1` to `Atr54`, `Class` column predicts the divorce, a value of `1` means couple would end up in divorce. You can read further about the dataset [here](https://dergipark.org.tr/en/pub/nevsosbilen/issue/46568/549416)

Download dataset from [here](https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set)


### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

As we have to predict either of two states (Divorce/No Divorce), this problem is a Classification one. The 54 features that we will use for prediction are described below. Each feature can have a value form the list `[0, 1, 2, 3, 4]`.

1. If one of us apologizes when our discussion deteriorates, the discussion ends.
2. I know we can ignore our differences, even if things get hard sometimes.
3. When we need it, we can take our discussions with my spouse from the beginning and correct it.
4. When I discuss with my spouse, to contact him will eventually work.
5. The time I spent with my wife is special for us.
6. We don't have time at home as partners.
7. We are like two strangers who share the same environment at home rather than family.
8. I enjoy our holidays with my wife.
9. I enjoy traveling with my wife.
10. Most of our goals are common to my spouse.
11. I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.
12. My spouse and I have similar values in terms of personal freedom.
13. My spouse and I have similar sense of entertainment.
14. Most of our goals for people (children, friends, etc.) are the same.
15. Our dreams with my spouse are similar and harmonious.
16. We're compatible with my spouse about what love should be.
17. We share the same views about being happy in our life with my spouse
18. My spouse and I have similar ideas about how marriage should be
19. My spouse and I have similar ideas about how roles should be in marriage
20. My spouse and I have similar values in trust.
21. I know exactly what my wife likes.
22. I know how my spouse wants to be taken care of when she/he sick.
23. I know my spouse's favorite food.
24. I can tell you what kind of stress my spouse is facing in her/his life.
25. I have knowledge of my spouse's inner world.
26. I know my spouse's basic anxieties.
27. I know what my spouse's current sources of stress are.
28. I know my spouse's hopes and wishes.
29. I know my spouse very well.
30. I know my spouse's friends and their social relationships.
31. I feel aggressive when I argue with my spouse.
32. When discussing with my spouse, I usually use expressions such as ‘you always’ or ‘you never’ .
33. I can use negative statements about my spouse's personality during our discussions.
34. I can use offensive expressions during our discussions.
35. I can insult my spouse during our discussions.
36. I can be humiliating when we discussions.
37. My discussion with my spouse is not calm.
38. I hate my spouse's way of open a subject.
39. Our discussions often occur suddenly.
40. We're just starting a discussion before I know what's going on.
41. When I talk to my spouse about something, my calm suddenly breaks.
42. When I argue with my spouse, ı only go out and I don't say a word.
43. I mostly stay silent to calm the environment a little bit.
44. Sometimes I think it's good for me to leave home for a while.
45. I'd rather stay silent than discuss with my spouse.
46. Even if I'm right in the discussion, I stay silent to hurt my spouse.
47. When I discuss with my spouse, I stay silent because I am afraid of not being able to control my anger.
48. I feel right in our discussions.
49. I have nothing to do with what I've been accused of.
50. I'm not actually the one who's guilty about what I'm accused of.
51. I'm not the one who's wrong about problems at home.
52. I wouldn't hesitate to tell my spouse about her/his inadequacy.
53. When I discuss, I remind my spouse of her/his inadequacy.
54. I'm not afraid to tell my spouse about her/his incompetence.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

The dataset has been uploaded into this github repository and it can be accessed using the link as below:

https://raw.githubusercontent.com/khalidw/Capstone-Project-Azure-Machine-Learning-Engineer/master/divorce.csv

We used method `from_delimited_files('webURL')` of the `TabularDatasetFactory` Class to retreive data from the csv file (link provided above).

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

Configuration and settings used for the Automated ML experiment are described in the table below:

Configuration | Description | Value
------------- | ----------- | -----
experiment_timeout_minutes | This is used as an exit criteria, it defines how long, in minutes, your experiment should continue to run | 20
max_concurrent_iterations | Represents the maximum number of iterations that would be executed in parallel | 5
primary_metric | The metric that Automated Machine Learning will optimize for model selection | accuracy
task | The type of task to run. Values can be 'classification', 'regression', or 'forecasting' depending on the type of automated ML problem | classification
compute_target | The compute target to run the experiment on | trainCluster
training_data | Training data, contains both features and label columns | ds
label_column_name | The name of the label column | Class
n_cross_validations | No. of cross validations to perform | 5 

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

In our experiment we found out `TruncatedSVDWrapper LightGBM` and `VotingEnsemble` to be the best model based on the accuracy metric. The accuracy score for these models was `1` i.e. `100%`. Since `100%` is the maximum accuracy a model can have, no further improvement can be made. However, improvements in terms of looking for another model can be done if this model is slow in inference.

The parameters for the model `TruncatedSVDWrapper LightGBM` are described in the table below.

`TruncatedSVDWrapper`

Parameters | Values
---------- | ------
n_components | 0.5047368421052632
random_state | None

`LightGBMClassifier`

Parameters | Values
---------- | ------
boosting_type | goss
class_weight | None
colsample_bytree | 0.5944444444444444
importance_type | split
learning_rate | 0.05263631578947369
max_bin | 360
max_depth | -1
min_child_samples | 9
min_child_weight | 0
min_split_gain | 0.3684210526315789
n_estimators | 400
n_jobs | 1
num_leaves | 89
objective | None
random_state | None
reg_alpha | 0.5263157894736842
reg_lambda | 0.5263157894736842
silent | True
subsample | 1
subsample_for_bin | 200000
subsample_freq | 0
verbose | -10

### AutoML Screenshots
*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters

**Run Details Widget**
![autoML_runDetails](Images/autoML_runDetails.png)

![autoML_runDetails_accuracy](Images/autoML_accuracy.png)

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

We use Logistric Regression algorithm from the SKLearn framework in conjuction with hyperDrive for hyperparameter tuning. There are two hyperparamters for this experiment, **C** and **max_iter**. **C** is the inverse regularization strength whereas max_iter is the maximum iteration to converge for the SKLearn Logistic Regression.

We have used random parameter sampling to sample over a discrete set of values. Random parameter sampling is great for discovery and getting hyperparameter combinations that you would not have guessed intuitively, although it often requires more time to execute.

The parameter search space used for **C** is `[1,2,3,4,5]` and for **max_iter** is `[80,100,120,150,170,200]`

The benchmark metric from model testing is then evaluated using hyperDrive early stopping policy. Execution of the pipeline is stopped if conditions specified by the policy are met.

We have used the [BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py). This policy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. This helps to improves computational efficiency.

For this experiment the configuratin used is; `evaluation_interval=1`, `slack_factor=0.2`, and `delay_evaluation=5`. This configration means that the policy would be applied to every `1*5` iteration of the pipeline and if `1.2*`value of the benchmark metric for current iteration is smaller than the best metric value so far, the run will be cancelled.

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

The highest accuracy that our Logistic Regression Model acheived was `0.9803921568627451`. The hyperparamteres that were used by this model are:

Hyperparameter | Value
-------------- | -----
Regularization Strength (C) | 4.0
Max Iterations (max_iter) | 80

### Hyperparameter Tuning Screenshots
*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

**Run Details Widget**

![hyperDrive_runDetails](Images/hyperDrive_runDetails.png)

![hyperDrive_runDetails1](Images/hyperDrive_runDetails1.png)

![hyperDrive_hyperParams](Images/hyperDrive_hyperParams.png)

**Best Model**

![hyperDrive_bestModel](Images/hyperDrive_bestModel.png)

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

## Citation
Yöntem, M , Adem, K , İlhan, T , Kılıçarslan, S. (2019). DIVORCE PREDICTION USING CORRELATION BASED FEATURE SELECTION AND ARTIFICIAL NEURAL NETWORKS. Nevşehir Hacı Bektaş Veli University SBE Dergisi, 9 (1), 259-273.
