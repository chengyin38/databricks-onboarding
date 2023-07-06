# Databricks notebook source
# MAGIC %md
# MAGIC ## Demo Library
# MAGIC
# MAGIC This [demo library](https://www.databricks.com/resources/demos/library?itm_data=demo_center) contains hands-on demo for public reference.

# COMMAND ----------

# MAGIC %pip install dbdemos

# COMMAND ----------

# MAGIC %md
# MAGIC Running the command below also automatically spins up a new cluster with the necessary configurations

# COMMAND ----------

import dbdemos
dbdemos.install("llm-dolly-chatbot")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Solution Accelerators
# MAGIC
# MAGIC [Solution accelerators](https://www.databricks.com/solutions/accelerators) are typically longer hands-on demos with a more "end-to-end" flavor. 
