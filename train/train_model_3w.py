from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = (
    SparkSession.builder
    .appName("TreinoModelo3W")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# =====================================
# 1) Ler o CSV real consolidado do Kaggle
# =====================================

df = (
    spark.read
         .option("header", True)
         .option("inferSchema", True)
         .csv("../data/petrobras_3w_dados_treinamento.csv")
)

# =====================================
# 2) Seleção de colunas úteis
# =====================================

FEATURES = [
    "P-TPT",
    "T-TPT",
    "P-MON-CKP",
    "T-JUS-CKP",
    "P-JUS-CKGL",
]

df = df.select(
    *[col(c).cast("double") for c in FEATURES],
    col("classe").cast("int").alias("label")
)

# Remove nulos
df = df.na.drop()

print("Total de linhas após limpeza:", df.count())

# =====================================
# 3) Treino / Teste
# =====================================

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Vetorização
assembler = VectorAssembler(
    inputCols=FEATURES,
    outputCol="features_vec"
)

# Normalização
scaler = StandardScaler(
    inputCol="features_vec",
    outputCol="features",
    withMean=True,
    withStd=True
)

# Classificador
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=120,
    maxDepth=12,
    seed=42
)

pipeline = Pipeline(stages=[assembler, scaler, rf])

# =====================================
# 4) Treinar
# =====================================

model = pipeline.fit(train_df)

# =====================================
# 5) Avaliar
# =====================================

pred = model.transform(test_df)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

f1 = evaluator.evaluate(pred)
acc = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(pred)

print("=================================")
print("Acurácia:", acc)
print("F1-score:", f1)
print("=================================")

# =====================================
# 6) Salvar modelo
# =====================================

MODEL_PATH = "../model/modelo_3w_rf"
model.write().overwrite().save(MODEL_PATH)

print(f"Modelo salvo em {MODEL_PATH}")

spark.stop()
