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
from pyspark.sql import DataFrame
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

# Caminho para salvar predições que serão usadas pelo dashboard
VIEW_PATH = os.path.join(BASE_DIR, "view", "predicoes")

# Garante que a pasta "view" exista (Spark cria "predicoes" internamente)
os.makedirs(os.path.dirname(VIEW_PATH), exist_ok=True)

print("BASE_DIR     =", BASE_DIR)
print("MODEL_PATH   =", MODEL_PATH)
print("CHECKPOINT   =", CHECKPOINT_PATH)
print("VIEW_PATH    =", VIEW_PATH)

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

spark.sparkContext.setLogLevel("ERROR")

# ==========================
# ESQUEMA DO JSON RECEBIDO
# ==========================
# Este schema deve refletir exatamente o que o producer envia.

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

# Seleciona as colunas que queremos exibir / salvar
output = predictions.select(
    "timestamp_envio",
    *features,
    "prediction",
    "classe_real",
)

# ==========================
# FUNÇÃO PARA SALVAR CADA MICRO-LOTE EM CSV (PARA O DASHBOARD)
# ==========================

def salvar_batch_csv(batch_df: DataFrame, batch_id: int):
    """
    Salva o micro-lote de previsões em CSV, cada batch em uma subpasta própria:
    view/predicoes/batch_<batch_id>/

    Isso evita sobrescrever os dados anteriores e permite acumular registros
    para visualização (matriz de confusão, gráficos, etc.).
    """
    if batch_df.rdd.isEmpty():
        return

    # Subpasta específica deste batch
    batch_path = os.path.join(VIEW_PATH, f"batch_{batch_id}")

    (
        batch_df
        .coalesce(1)  # 1 arquivo por batch
        .write
        .mode("overwrite")      # sobrescreve apenas a subpasta deste batch, não o diretório raiz
        .option("header", True)
        .csv(batch_path)
    )


# ==========================
# SAÍDAS DO STREAMING
# ==========================

# Stream 1: console (como antes)
query_console = (
    output.writeStream.outputMode("append")
    .format("console")
    .option("truncate", "false")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .start()
)

# Stream 2: grava o último batch em CSV na pasta "view/predicoes"
query_csv = (
    output.writeStream
    .outputMode("append")
    .foreachBatch(salvar_batch_csv)
    .start()
)

# Espera encerramento (normalmente via Ctrl+C)
query_console.awaitTermination()
query_csv.awaitTermination()
