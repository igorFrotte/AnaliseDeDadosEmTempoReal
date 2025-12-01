from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)
from pyspark.ml import PipelineModel

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "telemetry.raw"
MODEL_PATH = "../train/modelo_3w_rf"  

# 1) Sessão Spark com integração Kafka
spark = (
    SparkSession.builder
    .appName("KafkaSparkStreaming3W-Classification")
    .master("local[*]")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# 2) Schema do JSON vindo do producer
schema = StructType([
    StructField("P-TPT", DoubleType(), True),
    StructField("T-TPT", DoubleType(), True),
    StructField("P-MON-CKP", DoubleType(), True),
    StructField("T-JUS-CKP", DoubleType(), True),
    StructField("P-JUS-CKGL", DoubleType(), True),
    StructField("classe_real", IntegerType(), True),
    StructField("timestamp_envio", StringType(), True),
])

# 3) Ler o tópico Kafka
df_kafka = (
    spark.readStream
         .format("kafka")
         .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
         .option("subscribe", TOPIC)
         .option("startingOffsets", "latest")
         .load()
)

# 4) Converter VALUE (bytes) -> STRING -> JSON -> colunas
df_parsed = (
    df_kafka
    .selectExpr("CAST(value AS STRING) as json_str")
    .select(from_json(col("json_str"), schema).alias("data"))
    .select("data.*")
)

# 5) Limpeza / tipagem (em tese já veio como DoubleType pelo schema, mas garantimos)
features = ["P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP", "P-JUS-CKGL"]

df_clean = df_parsed
for f in features:
    df_clean = df_clean.withColumn(f, col(f).cast("double"))

df_clean = df_clean.na.drop(subset=features)

# 6) Carregar modelo treinado
model = PipelineModel.load(MODEL_PATH)

# 7) Aplicar o modelo no streaming
predictions = model.transform(df_clean)

# 8) Selecionar o que queremos mostrar
output = predictions.select(
    "timestamp_envio",
    *features,
    "prediction",
    "classe_real"
)

# 9) Escrever no console
query = (
    output
    .writeStream
    .outputMode("append")
    .format("console")
    .option("truncate", "false")
    .option("checkpointLocation", "/tmp/spark_checkpoint_3w_stream")
    .start()
)

query.awaitTermination()
