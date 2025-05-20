import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Mapear número a dataset
dataset_map = {
    "0": "small_celestial.csv",
    "1": "medium_celestial.csv",
    "2": "half_celestial.csv"
}

if len(sys.argv) != 2 or sys.argv[1] not in dataset_map:
    print("Uso: spark-submit code.py [0|1|2]")
    print("0 = small_celestial.csv")
    print("1 = medium_celestial.csv")
    print("2 = half_celestial.csv")
    sys.exit(1)

dataset_file = dataset_map[sys.argv[1]]
dataset_path = f"/datasets/{dataset_file}"

# Inicializa Spark
spark = SparkSession.builder.appName("ClasificacionCelestial").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Carga del conjunto de datos
df = spark.read.csv(dataset_path, header=True, sep=";", inferSchema=True)
print(f"\nUsando dataset: {dataset_path}")
df.printSchema()
df.show(5)

# 1. Análisis exploratorio
print("\nAnálisis exploratorio")
print("----------------------")
print(f"Número de filas: {df.count()}, Columnas: {len(df.columns)}")
df.describe().show()

print("Distribución de clases:")
df.groupBy("type").count().show()

print("Valores perdidos en cada columna:")
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# 2. Preprocesamiento
## Indexación de etiquetas
labelIndexer = StringIndexer(inputCol="type", outputCol="label").fit(df)
df = labelIndexer.transform(df)

## Balanceo mediante Submuestreo Aleatorio (RUS)
counts = df.groupBy("label").count().collect()
label_counts = {row['label']: row['count'] for row in counts}
min_count = min(label_counts.values())

print(f"\nBalanceando clases con submuestreo al tamaño de la clase minoritaria = {min_count}")

## Filtro por clase
dfs = []
for label_val in label_counts.keys():
    df_label = df.filter(col("label") == label_val)
    fraction = min_count / label_counts[label_val]
    if fraction < 1.0:
        df_sampled = df_label.sample(withReplacement=False, fraction=fraction, seed=42)
    else:
        df_sampled = df_label
    dfs.append(df_sampled)

df_balanced = dfs[0]
for i in range(1, len(dfs)):
    df_balanced = df_balanced.union(dfs[i])

print(f"Nuevo conteo balanceado por clase:")
df_balanced.groupBy("label").count().show()

## Ensamblado, escalado y normalización
input_cols = [c for c in df_balanced.columns if c != "type" and c != "label"]
assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
assembled = assembler.transform(df_balanced)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scaled = scaler.fit(assembled).transform(assembled)

normalizer = MinMaxScaler(inputCol="scaledFeatures", outputCol="normalizedFeatures")
normalized = normalizer.fit(scaled).transform(scaled)

# 3. Partición
train, test = normalized.randomSplit([0.8, 0.2], seed=1234)

# 4. Evaluación
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

## Logistic Regression
lr1 = LogisticRegression(featuresCol="normalizedFeatures", labelCol="label", maxIter=10, regParam=0.3)
lr2 = LogisticRegression(featuresCol="normalizedFeatures", labelCol="label", maxIter=20, regParam=0.1)

lr_model1 = lr1.fit(train)
lr_model2 = lr2.fit(train)

auc_lr1 = evaluator.evaluate(lr_model1.transform(test))
auc_lr2 = evaluator.evaluate(lr_model2.transform(test))

## Random Forest
rf1 = RandomForestClassifier(featuresCol="normalizedFeatures", labelCol="label", numTrees=10, maxDepth=5)
rf2 = RandomForestClassifier(featuresCol="normalizedFeatures", labelCol="label", numTrees=50, maxDepth=10)

rf_model1 = rf1.fit(train)
rf_model2 = rf2.fit(train)

auc_rf1 = evaluator.evaluate(rf_model1.transform(test))
auc_rf2 = evaluator.evaluate(rf_model2.transform(test))

## SVM
svm1 = LinearSVC(featuresCol="normalizedFeatures", labelCol="label", maxIter=10, regParam=0.1)
svm2 = LinearSVC(featuresCol="normalizedFeatures", labelCol="label", maxIter=20, regParam=0.01)

svm_model1 = svm1.fit(train)
svm_model2 = svm2.fit(train)

auc_svm1 = evaluator.evaluate(svm_model1.transform(test))
auc_svm2 = evaluator.evaluate(svm_model2.transform(test))

## Resultados
print("\nResultados obtenidos:")
print(f"AUC ROC para LR1: {auc_lr1}")
print(f"AUC ROC para LR2: {auc_lr2}")
print(f"AUC ROC para RF1: {auc_rf1}")
print(f"AUC ROC para RF2: {auc_rf2}")
print(f"AUC ROC para SVM1: {auc_svm1}")
print(f"AUC ROC para SVM2: {auc_svm2}")

# Finaliza Spark
spark.stop()