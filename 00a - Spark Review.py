# Databricks notebook source
# DBTITLE 0,--i18n-1108b110-983d-4034-9156-6b95c04dc62c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC # Spark Review
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC
# MAGIC * Create a Spark DataFrame and query it using Spark SQL
# MAGIC * Analyze the Spark UI
# MAGIC * Cache data with Spark and explain the advantages of data caching
# MAGIC * Convert Spark DataFrame to pandas DataFrame and vise versa

# COMMAND ----------

# DBTITLE 0,--i18n-890d085b-9058-49a7-aa15-bff3649b9e05
# MAGIC %md 
# MAGIC
# MAGIC ## Spark Architecture
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/sparkcluster.png)

# COMMAND ----------

# DBTITLE 0,--i18n-df081f79-6894-4174-a554-fa0943599408
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Spark DataFrame

# COMMAND ----------

from pyspark.sql.functions import col, rand

df = (spark.range(1, 1000000)
      .withColumn("id", (col("id") / 1000).cast("integer"))
      .withColumn("v", rand(seed=1)))

# COMMAND ----------

# DBTITLE 0,--i18n-a0c6912d-a8d6-449b-a3ab-5ca91c7f9805
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Why were no Spark jobs kicked off above? Well, we didn't have to actually "touch" our data, so Spark didn't need to execute anything across the cluster.

# COMMAND ----------

display(df.sample(.001))

# COMMAND ----------

# DBTITLE 0,--i18n-6eadef21-d75c-45ba-8d77-419d1ce0c06c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Views
# MAGIC
# MAGIC How can I access this in SQL?

# COMMAND ----------

df.createOrReplaceTempView("df_temp")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM df_temp LIMIT 10

# COMMAND ----------

# DBTITLE 0,--i18n-2593e6b0-d34b-4086-9fed-c4956575a623
# MAGIC %md 
# MAGIC
# MAGIC ## Count
# MAGIC
# MAGIC Let's see how many records we have.

# COMMAND ----------

df.count()

# COMMAND ----------

# DBTITLE 0,--i18n-5d00511e-15da-48e7-bd26-e89fbe56632c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Spark UI
# MAGIC
# MAGIC Open up the Spark UI - what are the shuffle read and shuffle write fields? The command below should give you a clue.

# COMMAND ----------

df.rdd.getNumPartitions()

# COMMAND ----------

# DBTITLE 0,--i18n-50330454-0168-4f50-8355-0204632b20ec
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Cache with Spark
# MAGIC
# MAGIC For repeated access, it will be much faster if we cache our data.

# COMMAND ----------

df.cache().count()

# COMMAND ----------

# DBTITLE 0,--i18n-7dd81880-1575-410c-a168-8ac081a97e9d
# MAGIC %md 
# MAGIC
# MAGIC **Re-run Count**
# MAGIC
# MAGIC Wow! Look at how much faster it is now!

# COMMAND ----------

df.count()

# COMMAND ----------

# DBTITLE 0,--i18n-ce238b9e-fee4-4644-9469-b7d9910f6243
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Collect Data
# MAGIC
# MAGIC When you pull data back to the driver  (e.g. call **`.collect()`**, **`.toPandas()`**,  etc), you'll need to be careful of how much data you're bringing back. Otherwise, you might get OOM exceptions!
# MAGIC
# MAGIC A best practice is explicitly limit the number of records, unless you know your data set is small, before calling **`.collect()`** or **`.toPandas()`**.

# COMMAND ----------

df.limit(10).toPandas()
