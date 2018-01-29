# Import data as dataframe
pyspark --packages com.databricks:spark-csv_2.10:1.2.0

from pyspark.sql import SQLContext
from pyspark.sql.types import *
sqlContext = SQLContext(sc)

df = sqlContext.read.load('hdfs:///user/training/mldata/titanic_train.csv,
                            format ='com.databricks.spark.csv',
                            head = 'true',
                            inferSchema= 'true')

# Print the dataset and verify that the schema contains all the variables.
df.show()
df.printSchema()

# Print the first 10 rows from the dataset
df.show(10)

# Obtain summary statistics for all variables in the dataframe.
#Pay attention to whether there are missing data as well as whether
#the field appears to be continuous or discrete.
df.describe().show()

# For each of the string columns (except name and ticket)
# print the count of the 10 most frequent values ordered by descending order of frequency.
df.groupBy('Cabin').count().orderBy('count', ascending=False).show(10)
df.groupBy('Embarked').count().orderBy('count', ascending=False).show(10)

# Select all feature columns you plan to use in addition to the
# target variable (i.e., ‘Survived’) and covert all numerical columns into double data type.
# Tip: you can use the .cast() from pyspark.sql.functions.

df_keep=df.select(df['Survived'].cast("double"),df['Pclass'].cast("double"),
                df['Sex'],df['Age'],df['SibSp'].cast("double"),
                df['Parch'].cast("double"),df['Fare'])

# Replace the missing values in the Age column with the mean value.
# Create also a new variable (e.g., ‘AgeNA’)
# indicating whether the value of age was missing or not.

udfNA = udf(NA, StringType())
df = df_keep.withColumn("AgeNA", udfNA("Age"))
df_rev = df.select(df['Survived'].cast("double"),df['Pclass'].cast("double"),
                df['Sex'],df['Age'],df['SibSp'].cast("double"),
                df['Parch'].cast("double"),df['Fare'],df['AgeNA'].cast("double"))
df_rev.groupBy().avg('Age').show()
# mean = 29.699118
df_rev = df_rev.na.fill({'Age': 29.699118})

# Print the revised dataframe and recalculate the summary statistics.
df_rev.show()
df_rev.describe().show()

# Encode all string and categorical variables in order to use them in the pipeline
# Import all necessary pyspark functions
# Create indexers and encoders for categorical string variables

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder

sex_indexer = StringIndexer(inputCol="Sex", outputCol="Sex_numeric").fit(df_rev)
indexed_df = sex_indexer.transform(df)

sex_encoder = OneHotEncoder(inputCol="Sex_numeric", outputCol="Sex_vector")
encoded_df = sex_encoder.transform(indexed_df)

# Assemble all feature columns into a feature vector in order to be used in the pipeline

assembler = VectorAssembler(
        inputCols=["Pclass", "Age", "SipSb", "Parch", "Fare", "AgeNA", "Sex_vector"],
        outputCol="features")

# Create the logistic regression model to be used in the pipeline

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter = 10, regParam = 0.01, featuresCol = 'features',
                        labelCol = 'Survived')

# Assemble the pipeline

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[sex_indexer, sex_enocder, assembler, lr])

# Use a 70-30 random split for the training and test sets, respectively

split = df_rev.randomSplit([0.7,0.3])
training = split[0]
test = split[1]

# Fit the model using the predefined pipeline on the training set.
# Use the fitted model for prediction on the test set.
# Report the logistic regression coefficients.

model = pipeline.fit(training)
pred = model.transform(test)
lrm = model.stages[-1]
lrm.coefficients

# Print the first 5 rows of the results.
# Report the AUC for this model.

pred.show(5)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
                                        labelCol='Survived')
evaluator.evaluate(pred)
#0.8493955485
