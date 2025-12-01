from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Criação da sessão Spark com integração Kafka
spark = SparkSession.builder \
    .appName("KafkaSparkStreaming3W") \
    .master("local[*]") \
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0"
    ) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Definição do esquema dos dados (mesmo formato do producer)
schema = StructType([
    StructField("well_id", StringType()),
    StructField("pressure", DoubleType()),
    StructField("temperature", DoubleType()),
    StructField("flow_rate", DoubleType()),
    StructField("timestamp", StringType())
])

# Ler o tópico Kafka em tempo real
df_kafka = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "telemetry.raw") \
    .option("startingOffsets", "latest") \
    .load()

# Converter valor binário para JSON
df_parsed = df_kafka.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Exibir o streaming em tempo real no console
query = df_parsed.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .option("checkpointLocation", "/tmp/spark_checkpoint_3w") \
    .start()

query.awaitTermination()
