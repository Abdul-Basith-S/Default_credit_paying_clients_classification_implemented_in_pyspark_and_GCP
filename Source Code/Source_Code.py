#!/usr/bin/env python
# coding: utf-8

# ### Creating spark session and loading dataset

# In[14]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName("Final_Project")     .config("spark.some.config.option", "some-value")     .getOrCreate()


# In[15]:


credit = spark.read.load("gs://bdpp_project/notebooks/jupyter/Project/ccdefault.csv",format="csv", sep=",", inferSchema="true", header="true")


# ### Data exploration

# #### Let's look at the schema of the data, see the total number of instances, look at the top 5 rows of the dataset and finaly see the summary of the whole dataset.

# In[16]:


credit.printSchema()


# In[17]:


credit.count()


# The dataset has a total of 30,000 instances/entries

# In[18]:


credit.take(5)


# Since it is quite difficult to interpret let's convert it to a pandas dataframe and look at the top 5 rows.

# In[19]:


credit.toPandas().head(n=5)


# In[20]:


credit.describe()


# In[21]:


credit.describe().toPandas().transpose()


# ###### From the summary and looking manually into the dataset I have infered three points:
# 
# 1. It can be observed that marriage has a lable '0' but in the description of data it is given marriage has values 1,2 and 3.
# 
# 2. Education column has 0, 5 and 6 as lables which are not specified in the decription of data.
# 
# 3. We can see that all the columns have -2 as minimum. But in the description of data it given that -1 indicates pay duly. There is no information regarding -2.
# 
# We can deal with these during data cleaning

# ### Let's see how different variables are distributed and how there are related to one another.

# 1. Check number of classes in the target variable is balanced of imbalenced.

# In[22]:


credit.groupBy('DEFAULT').count().show()


# Visualize it for better interpretation:

# In[23]:


#import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline
#credit_pd = credit.toPandas()
#print(len(credit_pd))
#plt.figure(figsize=(4,4))
#sns.countplot(x='DEFAULT', data=credit_pd, order=credit_pd['DEFAULT'].value_counts().index)


# 2. Let's check how target variable (DEFAULT) is related other technicaly categorical variables (SEX, EDUCATION and MARRIAGE)

# In[24]:


credit.select("SEX", "Default").groupBy("Default", 'SEX').count().show()


# In[25]:


credit.select("Marriage", "Default").groupBy("Default", 'Marriage').count().show()


# In[26]:


credit.select("Education", "Default").groupBy("Default", 'Education').count().show()


# 3. Let's see the correlations scores btw target variable and categorical variables:

# In[27]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["SEX","DEFAULT"],outputCol="S_D")
credit1 = assembler.transform(credit)
#credit1.select("S_D").show(5)

r1 = Correlation.corr(credit1, "S_D").head()
print("correlation matrix:\n" + str(r1[0]))


# In[28]:


assembler = VectorAssembler(inputCols=["EDUCATION","DEFAULT"],outputCol="E_D")
credit2 = assembler.transform(credit)

r2 = Correlation.corr(credit2, "E_D").head()
print("correlation matrix:\n" + str(r2[0]))


# In[29]:


assembler = VectorAssembler(inputCols=["MARRIAGE","DEFAULT"],outputCol="M_D")
credit2 = assembler.transform(credit)

r3 = Correlation.corr(credit2, "M_D").head()
print("correlation matrix:\n" + str(r3[0]))


# 4. I have plotted a heatmap to see the correlation btw all the attibutes/features.

# In[30]:


#import seaborn as sns; sns.set()
#correlation_df = credit.toPandas().corr()

#sns.set(rc={"font.style":"normal",
#            "axes.titlesize":20,
#            "text.color":"black",
#            "xtick.color":"black",
#            "ytick.color":"black",
#            "axes.labelcolor":"black",
#            "axes.grid":False,
#            'axes.labelsize':30,
#            'figure.figsize':(15, 15),
#            'xtick.labelsize':15,
#            'ytick.labelsize':15})
#sns.heatmap(correlation_df, annot = True, annot_kws={"size": 7}, cmap="YlGnBu", linewidths=.5)


# 5. Let's look at the correlation scores btw the target variable and all the other variables.

# In[31]:


import six
for i in credit.columns:
    if not( isinstance(credit.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to Default for ", i, credit.stat.corr('DEFAULT',i))


# ## Data cleaning

# Let us 1st fix the flaws we noted down earlier in the summary of the data.
# 
# 1. Assign '0' to other label in marriage column i.e., change all '0' to '3'.
# 2. Assign '0','5' & '6' to other label in education column i.e., change all '0','5' & '6' to '4'.
# 3. And I dont want negative values in Pay_x columns so I will change instances with '-2' to 0. Which means duly paid.
# 
# 

# In[32]:


from pyspark.sql.functions import *
from pyspark.sql.functions import when, count, col, countDistinct
credit.select(countDistinct('MARRIAGE')).show()


# In[33]:


from pyspark.sql import functions as F
credit_1 = credit.withColumn("MARRIAGE", F.when(F.col("MARRIAGE")==0, 3).otherwise(F.col("MARRIAGE")))
credit_1.select(countDistinct('MARRIAGE')).show()


# In[34]:


credit.select(countDistinct('EDUCATION')).show()


# In[35]:


credit_2 = credit_1.withColumn('EDUCATION', F.when(F.col("EDUCATION")==5, 4).otherwise(F.col("EDUCATION"))).withColumn('EDUCATION', F.when(F.col("EDUCATION")==6, 4).otherwise(F.col("EDUCATION"))).withColumn('EDUCATION', F.when(F.col("EDUCATION")==0, 4).otherwise(F.col("EDUCATION")))
credit_2.select(countDistinct('EDUCATION')).show()


# In[36]:


credit.select(countDistinct('PAY_0'),countDistinct('PAY_2'),countDistinct('PAY_3'),countDistinct('PAY_4'),countDistinct('PAY_5'),countDistinct('PAY_6')).show()


# In[37]:


credit_3 = credit_2.withColumn('PAY_0', F.when((F.col('PAY_0')==-2) | (F.col('PAY_0')==-1), 0).otherwise(F.col('PAY_0'))).withColumn('PAY_2', F.when((F.col('PAY_2')==-2) | (F.col('PAY_2')==-1), 0).otherwise(F.col('PAY_2'))).withColumn('PAY_3', F.when((F.col('PAY_3')==-2) | (F.col('PAY_3')==-1), 0).otherwise(F.col('PAY_3'))).withColumn('PAY_4', F.when((F.col('PAY_4')==-2) | (F.col('PAY_4')==-1), 0).otherwise(F.col('PAY_4'))).withColumn('PAY_5', F.when((F.col('PAY_5')==-2) | (F.col('PAY_5')==-1), 0).otherwise(F.col('PAY_5'))).withColumn('PAY_6', F.when((F.col('PAY_6')==-2) | (F.col('PAY_6')==-1), 0).otherwise(F.col('PAY_6')))
credit_3.select(countDistinct('PAY_0'),countDistinct('PAY_2'),countDistinct('PAY_3'),countDistinct('PAY_4'),countDistinct('PAY_5'),countDistinct('PAY_6')).show()


# ###### Check if there are any missing values in the dataset.

# In[38]:


credit.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in credit.columns]).toPandas()


# We don't have any missing values in the data set so we can move forward.

# ### Feature scaling

# In[39]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

assembler = VectorAssembler(inputCols=['ID','LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4',
'PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT1',
'PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'], outputCol="features")

featuredcredit = assembler.transform(credit_3)

scaler = StandardScaler(withMean=True, withStd=True, inputCol="features", outputCol="scaledFeatures")

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(featuredcredit)

# Normalize each feature to have unit standard deviation.
scaledcredit = scalerModel.transform(featuredcredit)
scaledcredit.select(["features", "scaledFeatures"]).show(5)


# In[40]:


scaledcredit.show(5)


# Creating final data set for training the model.

# In[41]:


renamedcredit = scaledcredit.withColumnRenamed("DEFAULT", "label")
Final_DS = renamedcredit.withColumn('features', renamedcredit.scaledFeatures).select("features","label")
Final_DS.show(5)


# ### Model training

# Dividing data into train and test sets:

# In[42]:


trainSet, testSet = Final_DS.randomSplit([0.8, 0.2], seed=12345)
print("Training Dataset Count: " + str(trainSet.count()))
print("Test Dataset Count: " + str(testSet.count()))


# ### Logistic Regression

# In[43]:


from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from time import *
start_time = time()

#Model training
lr = LogisticRegression(featuresCol = 'features',labelCol = 'label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(trainSet)
trainingSummary = lrModel.summary
trainaccuracy = trainingSummary.accuracy
#print("Coefficients: %s" % str(lrModel.coefficients))
#print("Intercept: %s" % str(lrModel.intercept))
print("Training accuracy for Logistic Regression: ",trainaccuracy)

end_time = time()
elapsed_time = end_time - start_time
print("Time to train Logistic Regression model: %.3f seconds" % elapsed_time)


# In[44]:


#Printing ROC value and ploting ROC curve
#train_roc = trainingSummary.roc.toPandas()
#plt.figure(figsize=(5,5))
#plt.plot(train_roc['FPR'],train_roc['TPR'])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#plt.show()
print("Train areaUnderROC for Logistic Regression: " + str(trainingSummary.areaUnderROC))


# In[45]:


start_time = time()

#Prediction
predictions_lr = lrModel.transform(testSet)
predictions_lr.select("features","probability","prediction", "label").show(5)
predictions_lr.groupBy('label','prediction').count().show()

end_time = time()
elapsed_time = end_time - start_time
print("Time for prediction Logistic Regression model: %.3f seconds" % elapsed_time)


# In[46]:


#Evaluating Logistic Regression Model.
start_time = time()

lr_evaluator = lrModel.evaluate(testSet)
lr_bc_evaluator = BinaryClassificationEvaluator()
lr_test_roc = lr_bc_evaluator.evaluate(predictions_lr)
print("Test areaUnderROC for Logistic Regression: ",lr_test_roc)


lr_testaccuracy = lr_evaluator.accuracy
print("Test accuracy for Logistic Regression: ",lr_testaccuracy)

end_time = time()
elapsed_time = end_time - start_time
print("Time to evaluate Logistic Regression model: %.3f seconds" % elapsed_time)


# ### Decision Tree

# In[47]:


from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

start_time = time()

#Model training
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(trainSet)

end_time = time()
elapsed_time = end_time - start_time
print("Time to train Decission Tree model: %.3f seconds" % elapsed_time)


# In[48]:


start_time = time()

#Prediction
predictions_dt = dtModel.transform(testSet)
predictions_dt.select("features","probability","prediction", "label").show(5)
predictions_dt.groupBy('label','prediction').count().show()

end_time = time()
elapsed_time = end_time - start_time
print("Time for prediction Decission Tree model: %.3f seconds" % elapsed_time)


# In[49]:


#Evaluating Decission Tree Model.

start_time = time()

dt_bc_evaluator = BinaryClassificationEvaluator()
dt_test_roc = dt_bc_evaluator.evaluate(predictions_dt, {dt_bc_evaluator.metricName: "areaUnderROC"})
dt_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
dt_testaccuracy = dt_evaluator.evaluate(predictions_dt)
print("Test_roc for Decission Tree:",dt_test_roc)
print("Test accuracy for Decisson Tree: ",dt_testaccuracy )

end_time = time()
elapsed_time = end_time - start_time
print("Time to evaluate Decission Tree model: %.3f seconds" % elapsed_time)


# ### Random Forest

# In[50]:


from pyspark.ml.classification import RandomForestClassifier

start_time = time()

#Training
rf = RandomForestClassifier(labelCol="label", featuresCol="features",numTrees=8)
rfModel = rf.fit(trainSet)

end_time = time()
elapsed_time = end_time - start_time
print("Time to train Random Forest model: %.3f seconds" % elapsed_time)


# In[51]:


start_time = time()

#prediction
predictions_rf = rfModel.transform(testSet)
predictions_rf.select("features","probability","prediction", "label").show(5)
predictions_rf.groupBy('label','prediction').count().show()

end_time = time()
elapsed_time = end_time - start_time
print("Time for prediction Random Forest model: %.3f seconds" % elapsed_time)


# In[52]:


#Evaluating Random Forest model

start_time = time()

rf_bc_evaluator = BinaryClassificationEvaluator()
rf_test_roc = rf_bc_evaluator.evaluate(predictions_rf, {rf_bc_evaluator.metricName: "areaUnderROC"})
rf_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
rf_testaccuracy = rf_evaluator.evaluate(predictions_rf)
print("Test_roc for Random Forest:",rf_test_roc)
print("Test accuracy for Random Forest: ",rf_testaccuracy )

end_time = time()
elapsed_time = end_time - start_time
print("Time to evaluate Random Forest model: %.3f seconds" % elapsed_time)


# ## Model Selection a.k.a. hyperparameter tuning

# ##### For LR

# In[53]:


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import BinaryClassificationMetrics

start_time = time()

# Creating ParamGrid for Cross Validation
lr_paramGrid = ParamGridBuilder()     .addGrid(lr.regParam, [0.1, 0.01])     .build()

# Creating CrossValidator
lr_crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=lr_paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=10)

# Run cross validations
lr_cvModel = lr_crossval.fit(trainSet)

end_time = time()
elapsed_time = end_time - start_time
print("Time to train Logistic Regression best model: %.3f seconds" % elapsed_time)

lr_cvModel.bestModel


# In[54]:


start_time = time()

#Performing predictions
predictions_lr = lr_cvModel.transform(testSet)
predictions_lr.select("features","probability", "prediction", "label").show(5)
predictions_lr.groupBy('label','prediction').count().show()

end_time = time()
elapsed_time = end_time - start_time
print("Time for predicting Logistic Regression best model: %.3f seconds" % elapsed_time)


# In[55]:


start_time = time()

#Evaluating Model
lr_cv_evaluator = BinaryClassificationEvaluator()
lr_test_roc = lr_cv_evaluator.evaluate(predictions_lr, {lr_cv_evaluator.metricName: "areaUnderROC"})
print("Test_roc for Logistic Regression best model:",lr_test_roc)

lr_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
lr_testaccuracy = lr_evaluator.evaluate(predictions_lr)
print("Test accuracy for Logistic Regression best model: ",lr_testaccuracy )

end_time = time()
elapsed_time = end_time - start_time
print("Time for evaluating Logistic Regression best model: %.3f seconds" % elapsed_time)


# ##### For RF

# In[56]:


start_time = time()

# Creating ParamGrid for Cross Validation
rf_paramGrid = ParamGridBuilder()   .addGrid(rf.maxDepth, [1, 2, 4, 5])   .addGrid(rf.minInstancesPerNode, [1, 2, 4, 5])   .build()

rf_cv_evaluator = BinaryClassificationEvaluator()

# Creating CrossValidator
rf_crossval = CrossValidator(estimator = rf,
                          estimatorParamMaps = rf_paramGrid,
                          evaluator = rf_cv_evaluator,
                          numFolds = 10)

# Run cross validations
rf_cvModel = rf_crossval.fit(trainSet)

end_time = time()
elapsed_time = end_time - start_time
print("Time to train Random Forest best model: %.3f seconds" % elapsed_time)

rf_cvModel.bestModel


# In[57]:


start_time = time()

#Performing predictions
predictions_rf = rf_cvModel.transform(testSet)
predictions_rf.select("features", "probability", "prediction","label").show(5)
predictions_rf.groupBy('label','prediction').count().show()

end_time = time()
elapsed_time = end_time - start_time
print("Time for predicting Random Forest best model: %.3f seconds" % elapsed_time)


# In[58]:


start_time = time()

#Evaluating model
rf_test_roc = rf_cv_evaluator.evaluate(predictions_rf, {rf_cv_evaluator.metricName: "areaUnderROC"})
print("Test_roc for Random Forest best model:",rf_test_roc)

rf_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
rf_testaccuracy = rf_evaluator.evaluate(predictions_rf)
print("Test accuracy for Random Forest best model: ",rf_testaccuracy )

end_time = time()
elapsed_time = end_time - start_time
print("Time for evaluating Random Forest best model: %.3f seconds" % elapsed_time)


# ### Working with imbalanced target variable
# Since the classes in our target variable is imbalanced we need to deal with so that the model is not biased to a single class.
# This can be delt by assigning weights to the classes.

# In[59]:


from pyspark.sql.functions import when
ratio = 0.85
def assigning_weights(default):
    return when(default == 1, ratio).otherwise(1*(1-ratio))
credit_4 = credit_3.withColumn('weights', assigning_weights(F.col('DEFAULT')))


# In[60]:


credit_4.select("weights").toPandas()


# ### Scaling features

# In[61]:


assembler = VectorAssembler(inputCols=['ID','LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4',
'PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT1',
'PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','weights'], outputCol="features")

featuredcredit = assembler.transform(credit_4)

scaler = StandardScaler(withMean=True, withStd=True, inputCol="features", outputCol="scaledFeatures")

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(featuredcredit)

# Normalize each feature to have unit standard deviation.
scaledcredit = scalerModel.transform(featuredcredit)
scaledcredit.select(["features", "scaledFeatures"]).show(5)


# ### Creating final dataset

# In[62]:


renamedcredit_1 = scaledcredit.withColumnRenamed("DEFAULT", "label")
Final_DS_1 = renamedcredit_1.withColumn('features', renamedcredit_1.scaledFeatures).select("features","label")
Final_DS_1.show(5)


# ### Spliting data for training and testing

# In[63]:


trainSet, testSet = Final_DS_1.randomSplit([0.8, 0.2], seed=12345)
print("Training Dataset Count: " + str(trainSet.count()))
print("Test Dataset Count: " + str(testSet.count()))


# ### Logistic Regression on balanced target variable data

# In[64]:


lr = LogisticRegression(featuresCol = 'features',labelCol = 'label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(trainSet)
trainingSummary = lrModel.summary
trainaccuracy = trainingSummary.accuracy
#print("Coefficients: %s" % str(lrModel.coefficients))
#print("Intercept: %s" % str(lrModel.intercept))
print("Training accuracy for Logistic Regression on weighted target variable: ",trainaccuracy)


#train_roc = trainingSummary.roc.toPandas()
#plt.figure(figsize=(5,5))
#plt.plot(train_roc['FPR'],train_roc['TPR'])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#plt.show()
print("Train areaUnderROC for Logistic Regression on weighted target variable: " + str(trainingSummary.areaUnderROC))

predictions = lrModel.transform(testSet)
predictions.select("features","probability","prediction", "label").show(5)


evaluator = lrModel.evaluate(testSet)
bc_evaluator = BinaryClassificationEvaluator()
test_roc = bc_evaluator.evaluate(predictions)
print("Test areaUnderROC for Logistic Regression on weighted target variable: ",test_roc)

testaccuracy = evaluator.accuracy
print("Test accuracy for Logistic Regression on weighted target variable: ",testaccuracy)


# ### Decission tree for balanced target variable data

# In[65]:


dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(trainSet)


predictions = dtModel.transform(testSet)
predictions.select("features","probability","prediction", "label").show(5)

bc_evaluator = BinaryClassificationEvaluator()
test_roc = bc_evaluator.evaluate(predictions, {bc_evaluator.metricName: "areaUnderROC"})
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
testaccuracy = evaluator.evaluate(predictions)
print("Test_roc for Decision tree on weighted target variable:",test_roc)
print("Test accuracy for Decision tree on weighted target variable: ",testaccuracy )


# ### Random forest for balanced target variable data

# In[66]:


rf = RandomForestClassifier(labelCol="label", featuresCol="features",numTrees=8)
rfModel = rf.fit(trainSet)

predictions = rfModel.transform(testSet)
predictions.select("features","probability","prediction", "label").show(5)

bc_evaluator = BinaryClassificationEvaluator()
test_roc = bc_evaluator.evaluate(predictions, {bc_evaluator.metricName: "areaUnderROC"})
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
testaccuracy = evaluator.evaluate(predictions)
print("Test_roc for Random forest on weighted target variable:",test_roc)
print("Test accuracy for Random forest on weighted target variable: ",testaccuracy )


# In[ ]:





# In[ ]:




