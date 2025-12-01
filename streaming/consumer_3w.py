import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)
from pyspark.ml import PipelineModel

# ==========================
# CONFIGURAÇÕES BÁSICAS
# ==========================

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "telemetry.raw"

# Descobre o diretório base do projeto a partir deste arquivo:
# .../3wProj/streaming/consumer_3w.py  -> BASE_DIR = .../3wProj
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Caminho absoluto do modelo treinado (pasta criada pelo train_model_3w.py)
MODEL_PATH = os.path.join(BASE_DIR, "train", "modelo_3w_rf")

# Caminho para o checkpoint do streaming (pasta será criada se não existir)
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint_3w_stream")

print("BASE_DIR     =", BASE_DIR)
print("MODEL_PATH   =", MODEL_PATH)
print("CHECKPOINT   =", CHECKPOINT_PATH)

# ==========================
# INICIALIZAÇÃO DO SPARK
# ==========================

spark = (
    SparkSession.builder
    .appName("KafkaSparkStreaming3W-Classification")
    .master("local[*]")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0",
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ==========================
# ESQUEMA DO JSON RECEBIDO
# ==========================
# Este schema DEVE refletir exatamente o que o producer envia.

schema = StructType(
    [
        StructField("P-TPT", DoubleType(), True),
        StructField("T-TPT", DoubleType(), True),
        StructField("P-MON-CKP", DoubleType(), True),
        StructField("T-JUS-CKP", DoubleType(), True),
        StructField("P-JUS-CKGL", DoubleType(), True),
        StructField("classe_real", IntegerType(), True),
        StructField("timestamp_envio", StringType(), True),
    ]
)

features = ["P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP", "P-JUS-CKGL"]

# ==========================
# LEITURA DO KAFKA
# ==========================

df_kafka = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

# value é um binário -> convertemos para string e depois para JSON estruturado
df_parsed = (
    df_kafka.selectExpr("CAST(value AS STRING) AS json_str")
    .select(from_json(col("json_str"), schema).alias("data"))
    .select("data.*")
)

# ==========================
# LIMPEZA / TIPAGEM
# ==========================

df_clean = df_parsed

# Garante que todas as features estejam como double (mesma tipagem do treino)
for f in features:
    df_clean = df_clean.withColumn(f, col(f).cast("double"))

# Descarta linhas com NaN em qualquer feature (compatível com o treino)
df_clean = df_clean.na.drop(subset=features)

# ==========================
# CARREGAMENTO DO MODELO
# ==========================

print("Carregando modelo de:", MODEL_PATH)
model = PipelineModel.load(MODEL_PATH)

# ==========================
# APLICAÇÃO DO MODELO NO STREAMING
# ==========================

predictions = model.transform(df_clean)

# Seleciona as colunas que queremos exibir
output = predictions.select(
    "timestamp_envio",
    *features,
    "prediction",
    "classe_real",
)

# ==========================
# SAÍDA NO CONSOLE
# ==========================

query = (
    output.writeStream.outputMode("append")
    .format("console")
    .option("truncate", "false")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .start()
)

query.awaitTermination()
