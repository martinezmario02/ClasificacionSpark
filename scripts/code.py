from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Inicializa Spark
spark = SparkSession.builder.appName("ClasificacionCelestial").getOrCreate()

# Carga del conjunto de datos con separador ;
df = spark.read.csv("/datasets/small_celestial.csv", header=True, sep=";", inferSchema=True)
df.printSchema()
df.show(5)

# Análisis exploratorio
print(f"Número de filas: {df.count()}, Columnas: {len(df.columns)}")
df.describe().show()
df.groupBy("type").count().show()
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# Preprocesamiento
labelIndexer = StringIndexer(inputCol="type", outputCol="label").fit(df)
df = labelIndexer.transform(df)

input_cols = [c for c in df.columns if c != "type" and c != "label"]

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
assembled = assembler.transform(df)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scaled = scaler.fit(assembled).transform(assembled)

train, test = scaled.randomSplit([0.8, 0.2], seed=1234)

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# Logistic Regression
lr1 = LogisticRegression(featuresCol="scaledFeatures", labelCol="label", maxIter=10, regParam=0.3)
lr2 = LogisticRegression(featuresCol="scaledFeatures", labelCol="label", maxIter=20, regParam=0.1)

lr_model1 = lr1.fit(train)
lr_model2 = lr2.fit(train)

auc_lr1 = evaluator.evaluate(lr_model1.transform(test))
auc_lr2 = evaluator.evaluate(lr_model2.transform(test))

# Random Forest
rf1 = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="label", numTrees=10, maxDepth=5)
rf2 = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="label", numTrees=50, maxDepth=10)

rf_model1 = rf1.fit(train)
rf_model2 = rf2.fit(train)

auc_rf1 = evaluator.evaluate(rf_model1.transform(test))
auc_rf2 = evaluator.evaluate(rf_model2.transform(test))

##Resultados
print("Resultados obtenidos:")

print(f"AUC ROC para LR1: {auc_lr1}")
print(f"AUC ROC para LR2: {auc_lr2}")

print(f"AUC ROC para RF1: {auc_rf1}")
print(f"AUC ROC para RF2: {auc_rf2}")

# Finaliza Spark
spark.stop()