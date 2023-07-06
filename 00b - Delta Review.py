# Databricks notebook source
# DBTITLE 0,--i18n-fd2d84ac-6a17-44c2-bb92-18b0c7fef797
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Delta Review
# MAGIC
# MAGIC There are a few key operations necessary to understand and make use of <a href="https://docs.delta.io/latest/quick-start.html#create-a-table" target="_blank">Delta Lake</a>. First, we are going to discuss why we need Delta Lake and its unique features. Then, we are going to create a Delta table and explain how Delta keeps track of data storage and updates. The final section is going to cover how Delta manages versioning which enables time travel. 
# MAGIC
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC
# MAGIC * Explain the architecture and benefits of Delta Lake
# MAGIC * Create a Delta table based on existing data
# MAGIC * Read and update data in a Delta table
# MAGIC * Access previous versions of your Delta Table using <a href="https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html" target="_blank">time travel</a>
# MAGIC * Explain how Delta Lake manages data versioning with <a href="https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html" target="_blank">transaction logs</a>
# MAGIC
# MAGIC
# MAGIC ### Dataset
# MAGIC
# MAGIC In this notebook, we will be using the  San Francisco Airbnb rental dataset from <a href="http://insideairbnb.com/get-the-data.html" target="_blank">Inside Airbnb</a>.

# COMMAND ----------

# DBTITLE 0,--i18n-6a1bb996-7b50-4f03-9bcd-3d3ec3069a6d
# MAGIC %md 
# MAGIC ## Lesson Setup
# MAGIC
# MAGIC The first thing we're going to do is to **run setup script**. This script will define the required configuration variables that are scoped to each user.
# MAGIC
# MAGIC Expected ML Runtime = 13.2 ML

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 0,--i18n-68fcecd4-2280-411c-94c1-3e111683c6a3
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Why Delta Lake?
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://user-images.githubusercontent.com/20408077/87175470-4d8e1580-c29e-11ea-8f33-0ee14348a2c1.png" width="500"/>
# MAGIC </div>
# MAGIC
# MAGIC At a glance, Delta Lake is an open-source storage layer that brings both **reliability and performance** to data lakes. Delta Lake provides ACID transactions, scalable metadata handling, and unifies streaming and batch data processing. 
# MAGIC
# MAGIC Delta Lake runs on top of your existing data lake and is fully compatible with Apache Spark APIs. <a href="https://docs.databricks.com/delta/delta-intro.html" target="_blank">For more information </a>

# COMMAND ----------

# DBTITLE 0,--i18n-8ce92b68-6e6c-4fd0-8d3c-a57f27e5bdd9
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Create and Query  a Delta Table
# MAGIC First we need to read the Airbnb dataset as a Spark DataFrame

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/"
airbnb_df = spark.read.format("parquet").load(file_path)

display(airbnb_df)

# COMMAND ----------

# DBTITLE 0,--i18n-c100b529-ac6b-4540-a3ff-4afa63577eee
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC The cell below converts the data to a Delta table using the schema provided by the Spark DataFrame.

# COMMAND ----------

# Converting Spark DataFrame to Delta Table
dbutils.fs.rm(DA.paths.working_dir, True)
airbnb_df.write.format("delta").mode("overwrite").save(DA.paths.working_dir)

# COMMAND ----------

# DBTITLE 0,--i18n-090a31f6-1082-44cf-8e2a-6c659ea796ea
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC A Delta directory can also be registered as a table in the metastore.

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DA.schema_name}")
spark.sql(f"USE {DA.schema_name}")

airbnb_df.write.format("delta").mode("overwrite").saveAsTable("delta_review")

# COMMAND ----------

# DBTITLE 0,--i18n-732577c2-095d-4278-8466-74e494a9c1bd
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Delta supports partitioning. Partitioning puts data with the same value for the partitioned column into its own directory. Operations with a filter on the partitioned column will only read directories that match the filter. This optimization is called partition pruning. Choose partition columns based in the patterns in your data, this dataset for example might benefit if partitioned by neighborhood.

# COMMAND ----------

airbnb_df.write.format("delta").mode("overwrite").partitionBy("neighbourhood_cleansed").option("overwriteSchema", "true").save(DA.paths.working_dir)

# COMMAND ----------

# DBTITLE 0,--i18n-e9ce863b-5761-4676-ae0b-95f3f5f027f6
# MAGIC %md 
# MAGIC
# MAGIC ## Understanding the Transaction Log
# MAGIC
# MAGIC <a href="https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html" target="_blank">Read more about Delta Transaction Logs.</a>
# MAGIC
# MAGIC Let's take a look at the Delta Transaction Log. We can see how Delta stores the different neighborhood partitions in separate files. Additionally, we can also see a directory called _delta_log.

# COMMAND ----------

display(dbutils.fs.ls(DA.paths.working_dir))

# COMMAND ----------

# DBTITLE 0,--i18n-ac970bba-1cf6-4aa3-91bb-74a797496eef
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://user-images.githubusercontent.com/20408077/87174138-609fe600-c29c-11ea-90cc-84df0c1357f1.png" width="500"/>
# MAGIC </div>
# MAGIC
# MAGIC When a user creates a Delta Lake table, that table’s transaction log is automatically created in the _delta_log subdirectory. As he or she makes changes to that table, those changes are recorded as ordered, atomic commits in the transaction log. Each commit is written out as a JSON file, starting with 000000.json. Additional changes to the table generate more JSON files.

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.working_dir}/_delta_log/"))

# COMMAND ----------

# DBTITLE 0,--i18n-2905b874-373b-493d-9084-8ff4f7583ccc
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Next, let's take a look at a Transaction Log File.
# MAGIC
# MAGIC The <a href="https://docs.databricks.com/delta/delta-utility.html" target="_blank">four columns</a> each represent a different part of the very first commit to the Delta Table where the table was created.<br><br>
# MAGIC
# MAGIC - The add column has statistics about the DataFrame as a whole and individual columns.
# MAGIC - The commitInfo column has useful information about what the operation was (WRITE or READ) and who executed the operation.
# MAGIC - The metaData column contains information about the column schema.
# MAGIC - The protocol version contains information about the minimum Delta version necessary to either write or read to this Delta Table.

# COMMAND ----------

display(spark.read.json(f"{DA.paths.working_dir}/_delta_log/00000000000000000000.json"))

# COMMAND ----------

# DBTITLE 0,--i18n-8f79d1df-d777-4364-9783-b52bc0eed81a
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC The second transaction log has 39 rows. This includes metadata for each partition. 

# COMMAND ----------

display(spark.read.json(f"{DA.paths.working_dir}/_delta_log/00000000000000000001.json"))

# COMMAND ----------

# DBTITLE 0,--i18n-18500df8-b905-4f24-957c-58040920d554
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Finally, let's take a look at the files inside one of the Neighborhood partitions. The file inside corresponds to the partition commit (file 01) in the _delta_log directory.

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.working_dir}/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# DBTITLE 0,--i18n-9f817cd0-87ec-457b-8776-3fc275521868
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Reading Data from Delta table

# COMMAND ----------

df = spark.read.format("delta").load(DA.paths.working_dir)
display(df)

# COMMAND ----------

# DBTITLE 0,--i18n-faba817b-7cbf-49d4-a32c-36a40f582021
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Updating Delta Table
# MAGIC
# MAGIC Let's filter for rows where the host is a superhost.

# COMMAND ----------

df_update = airbnb_df.filter(airbnb_df["host_is_superhost"] == "t")
display(df_update)

# COMMAND ----------

df_update.write.format("delta").mode("overwrite").save(DA.paths.working_dir)

# COMMAND ----------

df = spark.read.format("delta").load(DA.paths.working_dir)
display(df)

# COMMAND ----------

# DBTITLE 0,--i18n-e4cafdf4-a346-4729-81a6-fdea70f4929a
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's look at the files in the Bayview partition post-update. Remember, the different files in this directory are snapshots of your DataFrame corresponding to different commits.

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.working_dir}/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# DBTITLE 0,--i18n-25ca7489-8077-4b23-96af-8d801982367c
# MAGIC %md 
# MAGIC
# MAGIC ## Delta Time Travel

# COMMAND ----------

# DBTITLE 0,--i18n-c6f2e771-502d-46ed-b8d4-b02e3e4f4134
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Oops, actually we need the entire dataset! You can access a previous version of your Delta Table using <a href="https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html" target="_blank">Delta Time Travel</a>. Use the following two cells to access your version history. Delta Lake will keep a 30 day version history by default, though it can maintain that history for longer if needed.

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS train_delta;
# MAGIC CREATE TABLE train_delta USING DELTA LOCATION '${DA.paths.working_dir}'

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY train_delta

# COMMAND ----------

# DBTITLE 0,--i18n-61faa23f-d940-479c-95fe-5aba72c29ddf
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Using the **`versionAsOf`** option allows you to easily access previous versions of our Delta Table.

# COMMAND ----------

df = spark.read.format("delta").option("versionAsOf", 0).load(DA.paths.working_dir)
display(df)

# COMMAND ----------

# DBTITLE 0,--i18n-5664be65-8fd2-4746-8065-35ee8b563797
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC You can also access older versions using a timestamp.
# MAGIC
# MAGIC Replace the timestamp string with the information from your version history. Note that you can use a date without the time information if necessary.

# COMMAND ----------

# Use your own timestamp 
# time_stamp_string = "FILL_IN"

# OR programatically get the first verion's timestamp value
time_stamp_string = str(spark.sql("DESCRIBE HISTORY train_delta").collect()[-1]["timestamp"])

df = spark.read.format("delta").option("timestampAsOf", time_stamp_string).load(DA.paths.working_dir)
display(df)

# COMMAND ----------

# DBTITLE 0,--i18n-38e92ae2-b60a-428d-91c6-9ba622de156c
# MAGIC %md 
# MAGIC
# MAGIC ## Cleanup Log Files

# COMMAND ----------

# DBTITLE 0,--i18n-6cbe5204-fe27-438a-af54-87492c2563b5
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now that we're happy with our Delta Table, we can clean up our directory using **`VACUUM`**. Vacuum accepts a retention period in hours as an input.

# COMMAND ----------

# DBTITLE 0,--i18n-4da7827c-b312-4b66-8466-f0245f3787f4
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Uh-oh, our code doesn't run! By default, to prevent accidentally vacuuming recent commits, Delta Lake will not let users vacuum a period under 7 days or 168 hours. Once vacuumed, you cannot return to a prior commit through time travel, only your most recent Delta Table will be saved.
# MAGIC
# MAGIC Try changing the vacuum parameter to different values.

# COMMAND ----------

# from delta.tables import DeltaTable

# delta_table = DeltaTable.forPath(spark, DA.paths.working_dir)
# delta_table.vacuum(0)

# COMMAND ----------

# DBTITLE 0,--i18n-1150e320-5ed2-4a38-b39f-b63157bca94f
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC We can workaround this by setting a spark configuration that will bypass the default retention period check.

# COMMAND ----------

from delta.tables import DeltaTable

spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
delta_table = DeltaTable.forPath(spark, DA.paths.working_dir)
delta_table.vacuum(0)

# COMMAND ----------

# DBTITLE 0,--i18n-b845b2ea-2c11-4d6e-b083-d5908b65d313
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's take a look at our Delta Table files now. After vacuuming, the directory only holds the partition of our most recent Delta Table commit.

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.working_dir}/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# DBTITLE 0,--i18n-a7bcdad3-affb-4b00-b791-07c14f5e59d5
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Since vacuuming deletes files referenced by the Delta Table, we can no longer access past versions. The code below should throw an error.

# COMMAND ----------

# df = spark.read.format("delta").option("versionAsOf", 0).load(DA.paths.working_dir)
# display(df)
