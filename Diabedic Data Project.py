# Databricks notebook source
# MAGIC %md
# MAGIC #Loadding the data from the CSV file and presenting the data. 

# COMMAND ----------

import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
#Databricks initializes spark by default, so you can directly use.

schema = StructType([
        StructField("encounter_id", IntegerType(), True),
        StructField("patient_nbr", IntegerType(), True),
        StructField("rase", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("age", StringType(), True),
        StructField("weight", StringType(), True),
        StructField("admission_type_id", IntegerType(), True),
        StructField("discharge_disposition_id", IntegerType(), True),
        StructField("admission_source_id", IntegerType(), True),
        StructField("time_in_hospital", IntegerType(), True),
        StructField("payer_code", StringType(), True),
        StructField("medical_specialty", StringType(), True),
        StructField("num_lab_procedures", IntegerType(), True),
        StructField("num_procedures", IntegerType(), True),
        StructField("num_medications", IntegerType(), True),
        StructField("number_outpatient", IntegerType(), True),
        StructField("number_emergency", IntegerType(), True),
        StructField("number_inpatient", IntegerType(), True),
        StructField("diag_1", StringType(), True),
        StructField("diag_2", StringType(), True),
        StructField("diag_3", StringType(), True),
        StructField("number_diagnoses", IntegerType(), True),
        StructField("max_glu_serum", StringType(), True),
        StructField("A1Cresult", StringType(), True),
        StructField("metformin", StringType(), True),
        StructField("repaglinide", StringType(), True),
        StructField("nateglinide", StringType(), True),
        StructField("chlorpropamide", StringType(), True),
        StructField("glimepiride", StringType(), True),
        StructField("acetohexamide", StringType(), True),
        StructField("glipizide", StringType(), True),
        StructField("glyburide", StringType(), True),
        StructField("tolbutamide", StringType(), True),
        StructField("pioglitazone", StringType(), True),
        StructField("rosiglitazone", StringType(), True),
        StructField("acarbose", StringType(), True),
        StructField("miglitol", StringType(), True),
        StructField("troglitazone", StringType(), True),
        StructField("tolazamide", StringType(), True),
        StructField("examide", StringType(), True),
        StructField("citoglipton", StringType(), True),
        StructField("insulin", StringType(), True),
        StructField("glyburide-metformin", StringType(), True),
        StructField("glipizide-metformin", StringType(), True),
        StructField("glimepiride-pioglitazone", StringType(), True),
        StructField("metformin-rosiglitazone", StringType(), True),
        StructField("metformin-pioglitazone", StringType(), True),
        StructField("change", StringType(), True),
        StructField("diabetesMed", StringType(), True),
        StructField("readmitted", StringType(), True),
         # Add more columns as needed
    ])

# Workspace path (not DBFS)
csv_path = "/Workspace/Users/enestrovelasco73@gmail.com/diabetic_data.csv"

# Read CSV with Pandas
df_pandas = pd.read_csv(csv_path)

# Convert to Spark DataFrame 
df_spark = spark.createDataFrame(df_pandas)

# Load the CSV file into a DataFrame
df = spark.read.csv(csv_path, header=True, inferSchema=True)


# Show data
#df.show(5)
df_spark.show(5)
df_pandas.head(5)
df_spark.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #Handling Missing Values ("?")

# COMMAND ----------

from pyspark.sql.functions import when, col

# Columns to check for "?"
columns_to_clean = ["race", "gender", "diag_1", "diag_2", "diag_3"]

# Replace "?" with NULL
for column in columns_to_clean:
    df = df.withColumn(column, 
                      when(col(column) == "?", None)
                      .otherwise(col(column)))

# COMMAND ----------

# MAGIC %md
# MAGIC #Check Missing Values After Cleaning

# COMMAND ----------

from pyspark.sql.functions import count, isnull

# Count NULLs per column
df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Impute Missing Values

# COMMAND ----------

from pyspark.ml.feature import Imputer

#Impute "race" with mode (most frequent category)
df.groupBy("rase").count().orderBy("count", ascending=False).show()  # Check mode

df = df.na.fill({"race": "Caucasian"})  # Replace NULLs with "Caucasian"

# COMMAND ----------

# MAGIC %md
# MAGIC # Handling Invalid Entries 

# COMMAND ----------

# Check current gender distribution
df.groupBy("gender").count().show()

#Remove "Unknown/Invalid" entries
df_cleaned = df.filter(col("gender") != "Unknown/Invalid")

#Replace "Unknown/Invalid" with "Other"
df = df.withColumn("gender", 
                  when(col("gender") == "Unknown/Invalid", "Other")
                  .otherwise(col("gender")))

# COMMAND ----------

# MAGIC %md
# MAGIC #Convert Categorical Columns to Meaningful Values

# COMMAND ----------

from pyspark.sql.functions import col, create_map, lit
from itertools import chain

# Define mapping dictionaries
admission_type_map = {
    1: "Emergency", 
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Unknown"
}

# Convert dictionary to Spark map type
mapping_expr = create_map([lit(x) for x in chain(*admission_type_map.items())])

# Apply mapping
df = spark.createDataFrame([], schema).withColumn("admission_type_desc", mapping_expr[col("admission_type_id")])



# COMMAND ----------

# MAGIC %md
# MAGIC #Calculate total_visits

# COMMAND ----------

from pyspark.sql.functions import coalesce, lit

df = df.withColumn("total_visits",
                  coalesce(col("number_outpatient"), lit(0)) + 
                  coalesce(col("number_emergency"), lit(0)) + 
                  coalesce(col("number_inpatient"), lit(0)))

# COMMAND ----------

# MAGIC %md
# MAGIC #Create readmission_flag

# COMMAND ----------

from pyspark.sql.functions import when

df = df.withColumn("readmission_flag",
                  when((col("readmitted") == "<30") | (col("readmitted") == ">30"), 1)
                  .otherwise(0))

# COMMAND ----------

# MAGIC %md
# MAGIC #Verifying Transformations

# COMMAND ----------

# Show sample of transformed data
df.select("admission_type_id", "admission_type_desc",
         "number_outpatient", "number_emergency", "number_inpatient", "total_visits",
         "readmitted", "readmission_flag").show(10)

# Check value counts
df.groupBy("readmission_flag").count().show()
#df.groupBy("age_group").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Saving Transformed Data

# COMMAND ----------

# Save to processed location
df.write.mode("overwrite").parquet("dbfs:/Users/enestrovelasco73@gmail.com/diabetic_data_processed")

# COMMAND ----------

# MAGIC %md
# MAGIC #Average Time in Hospital per Age Group or Diagnosis

# COMMAND ----------

from pyspark.sql.functions import avg, round

# By age group
avg_time_by_age = df.groupBy("age_group") \
                   .agg(round(avg("time_in_hospital"), 2).alias("avg_hospital_days")) \
                   .orderBy("avg_hospital_days", ascending=False)

print("Average Hospital Stay by Age Group:")
avg_time_by_age.show()

# By diagnosis category
avg_time_by_diag = df.groupBy("diag_1_category") \
                    .agg(round(avg("time_in_hospital"), 2).alias("avg_hospital_days")) \
                    .orderBy("avg_hospital_days", ascending=False)

print("\nAverage Hospital Stay by Diagnosis Category:")
avg_time_by_diag.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Diagnoses with High Readmission Rates

# COMMAND ----------

from pyspark.sql.functions import count, sum

# Calculate readmission rates by diagnosis
readmission_rates = df.groupBy("diag_1_category") \
                     .agg(
                         count("*").alias("total_cases"),
                         sum("readmission_flag").alias("readmitted_cases"),
                         (sum("readmission_flag")/count("*")*100).alias("readmission_rate")
                     ) \
                     .orderBy("readmission_rate", ascending=False)

print("Diagnoses with Highest Readmission Rates:")
readmission_rates.show()

# Filter for significant results (e.g., >100 cases)
print("\nSignificant Diagnoses (100+ cases):")
readmission_rates.filter(col("total_cases") > 100).show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Insulin/A1C Levels and Readmission Relationship

# COMMAND ----------

from pyspark.sql.functions import count

# Insulin analysis
insulin_readmission = df.groupBy("insulin", "readmission_flag") \
                       .agg(count("*").alias("count")) \
                       .orderBy("insulin", "readmission_flag")

print("Readmission by Insulin Status:")
insulin_readmission.show()

# A1C analysis (assuming A1C results are available)
if "A1Cresult" in df.columns:
    a1c_readmission = df.groupBy("A1Cresult", "readmission_flag") \
                       .agg(count("*").alias("count")) \
                       .orderBy("A1Cresult", "readmission_flag")
    
    print("\nReadmission by A1C Results:")
    a1c_readmission.show()
else:
    print("\nA1Cresult column not found in dataset")

# Statistical test example (using chi-square)
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# Prepare data for insulin analysis
assembler = VectorAssembler(
    inputCols=["insulin_index"],  # Need to convert to numeric index first
    outputCol="features"
)

pipeline = Pipeline(stages=[assembler])
model = pipeline.fit(df)
data = model.transform(df)

# Run chi-square test
r = ChiSquareTest.test(data, "features", "readmission_flag").head()
print(f"\nChi-square p-value for insulin relationship: {r.pValues[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Partition by Readmission Flag

# COMMAND ----------

df.write \
  .partitionBy("readmission_flag") \
  .mode("overwrite") \
  .parquet("/Workspace/Users/enestrovelasco73@gmail.com/diabetic_data_partitioned_by_readmission")

# COMMAND ----------

# MAGIC %md
# MAGIC #Cache Frequently Used DataFrames

# COMMAND ----------

# Cache the cleaned base DataFrame
df.cache()
print(f"Base DataFrame cached: {df.is_cached}")

# Cache aggregated results
avg_time_by_age.cache()
readmission_rates.cache()
print(f"Avg time by age cached: {avg_time_by_age.is_cached}")
print(f"Readmission rates cached: {readmission_rates.is_cached}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Optimized Analysis Pipeline

# COMMAND ----------

from pyspark.sql.functions import broadcast

def optimized_analysis():
    # Load partitioned data (example for age)
    age_partitioned = spark.read.parquet(
        "/Workspace/Users/enestrovelasco73@gmail.com/diabetic_data_partitioned_by_age"
    )
    
    # Cache partitioned data
    age_partitioned = age_partitioned.filter(col("age_group") == "Senior").cache()
    
    # Broadcast small lookup tables
    diag_lookup = spark.table("diagnosis_codes")  # Assuming exists
    broadcast_diag = broadcast(diag_lookup)
    
    # Join with broadcast
    enhanced_data = age_partitioned.join(
        broadcast_diag,
        age_partitioned.diag_1 == broadcast_diag.code,
        "left"
    )
    
    # Cache final enhanced data
    enhanced_data.cache()
    
    return enhanced_data

# Run optimized analysis
enhanced_df = optimized_analysis()

# COMMAND ----------

# MAGIC %md
# MAGIC #Monitoring Cache Usage

# COMMAND ----------

# Check storage memory used
storage_level = df.storageLevel
print(f"Storage Level: {storage_level}")
print("Memory Used: Access to sparkContext is not supported in serverless compute.")

# Clear cache when done
df.unpersist()
avg_time_by_age.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC #Save as Parquet

# COMMAND ----------

# Save as Parquet (columnar format, compressed)
clean_data_path = "/Workspace/Users/enestrovelasco73@gmail.com/diabetic_data_clean.parquet"

df.write \
  .mode("overwrite") \
  .option("compression", "snappy") \  # Good balance of speed/size
  .parquet(clean_data_path)

print(f"Cleaned data saved as Parquet to: {clean_data_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Export Key Metrics for BI Tools

# COMMAND ----------

from pyspark.sql.functions import col

# 1. Readmission Rates by Diagnosis
readmission_by_diag = df.groupBy("diag_1_category") \
                      .agg({
                          "readmission_flag": "avg",
                          "encounter_id": "count"
                      }) \
                      .withColumnRenamed("avg(readmission_flag)", "readmission_rate") \
                      .withColumnRenamed("count(encounter_id)", "case_count") \
                      .filter(col("case_count") > 100) \
                      .orderBy("readmission_rate", ascending=False)

# Save as single CSV file (coalesce to 1 partition)
readmission_by_diag.coalesce(1) \
  .write \
  .mode("overwrite") \
  .option("header", "true") \
  .csv("/Workspace/Users/enestrovelasco73@gmail.com/readmission_by_diagnosis")

# 2. Hospital Stay Duration by Age Group
stay_by_age = df.groupBy("age_group") \
               .agg({
                   "time_in_hospital": "avg",
                   "encounter_id": "count"
               }) \
               .withColumnRenamed("avg(time_in_hospital)", "avg_stay_days") \
               .withColumnRenamed("count(encounter_id)", "case_count")

stay_by_age.coalesce(1) \
  .write \
  .mode("overwrite") \
  .option("header", "true") \
  .csv("/Workspace/Users/enestrovelasco73@gmail.com/hospital_stay_by_age")

# 3. Medication Impact Analysis (Example)
if "insulin" in df.columns:
    med_impact = df.groupBy("insulin") \
                 .agg({
                     "readmission_flag": "avg",
                     "time_in_hospital": "avg"
                 })
    
    med_impact.coalesce(1) \
      .write \
      .mode("overwrite") \
      .option("header", "true") \
      .csv("/Workspace/Users/enestrovelasco73@gmail.com/medication_impact")