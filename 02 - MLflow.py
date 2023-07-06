# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-b27f81af-5fb6-4526-b531-e438c0fda55e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # MLflow
# MAGIC
# MAGIC <a href="https://mlflow.org/docs/latest/concepts.html" target="_blank">MLflow</a> seeks to address these three core issues:
# MAGIC
# MAGIC * It’s difficult to keep track of experiments
# MAGIC * It’s difficult to reproduce code
# MAGIC * There’s no standard way to package and deploy models
# MAGIC
# MAGIC In the past, when examining a problem, you would have to manually keep track of the many models you created, as well as their associated parameters and metrics. This can quickly become tedious and take up valuable time, which is where MLflow comes in.
# MAGIC
# MAGIC MLflow is pre-installed on the Databricks Runtime for ML.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC
# MAGIC * Utilize MLflow to track experiments and log metrics
# MAGIC * Query and view past runs programmatically
# MAGIC * Search and view past runs using MLflow UI
# MAGIC * Save and reload models using MLflow

# COMMAND ----------

# DBTITLE 0,--i18n-1e2c921e-1125-4df3-b914-d74bf7a73ab5
# MAGIC %md 
# MAGIC ## 📌 Requirements
# MAGIC
# MAGIC **Required Databricks Runtime Version:** 
# MAGIC * Please note that in order to run this notebook, you must use one of the following Databricks Runtime(s): **13.2.x-cpu-ml-scala2.12**

# COMMAND ----------

# DBTITLE 0,--i18n-6a1bb996-7b50-4f03-9bcd-3d3ec3069a6d
# MAGIC %md 
# MAGIC ## Lesson Setup
# MAGIC
# MAGIC The first thing we're going to do is to **run setup script**. This script will define the required configuration variables that are scoped to each user.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 0,--i18n-b7c8a0e0-649e-4814-8310-ae6225a57489
# MAGIC %md 
# MAGIC
# MAGIC ## MLflow Architecture
# MAGIC
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# DBTITLE 0,--i18n-c1a29688-f50a-48cf-9163-ebcc381dfe38
# MAGIC %md 
# MAGIC
# MAGIC ## Load Dataset
# MAGIC
# MAGIC Let's start by loading in our SF Airbnb Dataset.

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)
print(train_df.cache().count())

# COMMAND ----------

# DBTITLE 0,--i18n-9ab8c080-9012-4f38-8b01-3846c1531a80
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## MLflow Tracking
# MAGIC
# MAGIC MLflow Tracking is a logging API specific for machine learning and agnostic to libraries and environments that do the training.  It is organized around the concept of **runs**, which are executions of data science code.  Runs are aggregated into **experiments** where many runs can be a part of a given experiment and an MLflow server can host many experiments.
# MAGIC
# MAGIC You can use <a href="https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment" target="_blank">mlflow.set_experiment()</a> to set an experiment, but if you do not specify an experiment, it will automatically be scoped to this notebook.

# COMMAND ----------

# DBTITLE 0,--i18n-82786653-4926-4790-b867-c8ccb208b451
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Track Runs
# MAGIC
# MAGIC Each run can record the following information:<br><br>
# MAGIC
# MAGIC - **Parameters:** Key-value pairs of input parameters such as the number of trees in a random forest model
# MAGIC - **Metrics:** Evaluation metrics such as RMSE or Area Under the ROC Curve
# MAGIC - **Artifacts:** Arbitrary output files in any format.  This can include images, pickled models, and data files
# MAGIC - **Source:** The code that originally ran the experiment
# MAGIC
# MAGIC **NOTE**: For Spark models, MLflow can only log PipelineModels.

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

with mlflow.start_run(run_name="LR-Single-Feature") as run:
    # Define pipeline
    vec_assembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")
    lr = LinearRegression(featuresCol="features", labelCol="price")
    pipeline = Pipeline(stages=[vec_assembler, lr])
    pipeline_model = pipeline.fit(train_df)

    # Log parameters
    mlflow.log_param("label", "price")
    mlflow.log_param("features", "bedrooms")

    # Log model
    mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas()) 

    # Evaluate predictions
    pred_df = pipeline_model.transform(test_df)
    regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
    rmse = regression_evaluator.evaluate(pred_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)

# COMMAND ----------

# DBTITLE 0,--i18n-70188282-8d26-427d-b374-954e9a058000
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Finally, we will use Linear Regression to predict the log of the price, due to its log normal distribution. 
# MAGIC
# MAGIC We'll also practice logging artifacts to keep a visual of our log normal histogram.

# COMMAND ----------

from pyspark.sql.functions import col, log, exp
import matplotlib.pyplot as plt

with mlflow.start_run(run_name="LR-Log-Price") as run:
    # Take log of price
    log_train_df = train_df.withColumn("log_price", log(col("price")))
    log_test_df = test_df.withColumn("log_price", log(col("price")))

    # Log parameter
    mlflow.log_param("label", "log_price")
    mlflow.log_param("features", "all_features")

    # Create pipeline
    r_formula = RFormula(formula="log_price ~ . - price", featuresCol="features", labelCol="log_price", handleInvalid="skip")  
    lr = LinearRegression(labelCol="log_price", predictionCol="log_prediction")
    pipeline = Pipeline(stages=[r_formula, lr])
    pipeline_model = pipeline.fit(log_train_df)

    # Log model
    mlflow.spark.log_model(pipeline_model, "log-model", input_example=log_train_df.limit(5).toPandas())

    # Make predictions
    pred_df = pipeline_model.transform(log_test_df)
    exp_df = pred_df.withColumn("prediction", exp(col("log_prediction")))

    # Evaluate predictions
    rmse = regression_evaluator.setMetricName("rmse").evaluate(exp_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(exp_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Log artifact
    plt.clf()

    log_train_df.toPandas().hist(column="log_price", bins=100)
    fig = plt.gcf()
    mlflow.log_figure(fig, f"{DA.username}_log_normal.png")
    plt.show()

# COMMAND ----------

# DBTITLE 0,--i18n-66785d5e-e1a7-4896-a8a9-5bfcd18acc5c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC That's it! Now, let's use MLflow to easily look over our work and compare model performance. You can either query past runs programmatically or use the MLflow UI.

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow LLM Tracking
# MAGIC
# MAGIC https://mlflow.org/docs/latest/python_api/mlflow.llm.html#mlflow.llm.log_predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow LLM Evaluation
# MAGIC
# MAGIC https://www.databricks.com/blog/announcing-mlflow-24-llmops-tools-robust-model-evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow PromptLab
# MAGIC Head to E2 Dogfood to test out MLflow Prompt Evaluation tool! 

# COMMAND ----------



# COMMAND ----------

# DBTITLE 0,--i18n-0b1a68e1-bd5d-4f78-a452-90c7ebcdef39
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### BONUS: Querying Past Runs Programmatically
# MAGIC
# MAGIC You can query past runs programmatically in order to use this data back in Python.  The pathway to doing this is an **`MlflowClient`** object.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

display(client.search_experiments())

# COMMAND ----------

# DBTITLE 0,--i18n-dcd771b2-d4ed-4e9c-81e5-5a3f8380981f
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC You can also use `search_runs` <a href="https://mlflow.org/docs/latest/search-syntax.html" target="_blank">(documentation)</a> to find all runs for a given experiment.

# COMMAND ----------

experiment_id = run.info.experiment_id
runs_df = mlflow.search_runs(experiment_id)

display(runs_df)

# COMMAND ----------

# DBTITLE 0,--i18n-68990866-b084-40c1-beee-5c747a36b918
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Pull the last run and look at metrics.

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
runs[0].data.metrics

# COMMAND ----------

runs[0].info.run_id

# COMMAND ----------

# DBTITLE 0,--i18n-cfbbd060-6380-444f-ba88-248e10a56559
# MAGIC %md 
# MAGIC
# MAGIC ### View past runs using MLflow UI
# MAGIC
# MAGIC Go to the **"Experiments"** page and click on the experiment that you want to view. 
# MAGIC
# MAGIC Examine the following experiment details using the UI:<br><br>
# MAGIC * The **`Experiment ID`**
# MAGIC * The artifact location.  This is where the artifacts are stored in DBFS.
# MAGIC
# MAGIC #### Table View
# MAGIC
# MAGIC You can customize the table view which lists all runs for the experiment. For example, you can show/hide `rmse` or `features` columns.
# MAGIC
# MAGIC Following details can be found on the Experiment list page:
# MAGIC * **Run Name**: This is the run name is used while logging the run. Click on the name to view details of the run. See steps below for more details about run page.
# MAGIC * **Duration**: This shows the elapsed time for each run.
# MAGIC * **Source**: This is the notebook that created this run.
# MAGIC * **Model**: This column shows the model type.
# MAGIC
# MAGIC
# MAGIC After clicking on the time of the run, take a look at the following:<br><br>
# MAGIC * The Run ID will match what we printed above
# MAGIC * The model that we saved, included a pickled version of the model as well as the Conda environment and the **`MLmodel`** file.
# MAGIC
# MAGIC Note that you can add notes under the "Notes" tab to help keep track of important information about your models. 
# MAGIC
# MAGIC Also, click on the run for the log normal distribution and see that the histogram is saved in "Artifacts".
# MAGIC
# MAGIC
# MAGIC #### Chart View
# MAGIC
# MAGIC Chart view allows you to compare runs by features and evaluation metric. You can use various charts, such as bar chart or scatter plot chart, to visually compare experiment runs.

# COMMAND ----------

# DBTITLE 0,--i18n-63ca7584-2a86-421b-a57e-13d48db8a75d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Load Saved Model
# MAGIC
# MAGIC Let's practice <a href="https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html" target="_blank">loading</a> our logged log-normal model.

# COMMAND ----------

model_path = f"runs:/{run.info.run_id}/log-model"
loaded_model = mlflow.spark.load_model(model_path)

display(loaded_model.transform(test_df))

# COMMAND ----------

# DBTITLE 0,--i18n-a2c7fb12-fd0b-493f-be4f-793d0a61695b
# MAGIC %md 
# MAGIC
# MAGIC ## Classroom Cleanup
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson:

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
