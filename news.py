from pyspark.ml import Pipeline 
from pyspark.ml.feature import CountVectorizer,StringIndexer, RegexTokenizer,StopWordsRemover
from pyspark.sql.functions import col, udf,regexp_replace,isnull
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StringType,IntegerType
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
import datetime

if __name__ == "__main__":
    time1 = datetime.datetime.now()
    spark = SparkSession\
        .builder\
        .appName("news")\
        .config("spark.some.config.option", "some-value")\
        .getOrCreate()
    bucket_path = sys.argv[1]
    news_data = spark.read.csv(bucket_path,header = 'True',inferSchema='True')
    title_category = news_data.select("TITLE","CATEGORY")
    title_category = title_category.dropna()
    title_category = title_category.withColumn("only_str",regexp_replace(col('TITLE'), '\d+', ''))
    regex_tokenizer = RegexTokenizer(inputCol="only_str", outputCol="words", pattern="\\W")
    raw_words = regex_tokenizer.transform(title_category)
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    words_df = remover.transform(raw_words)
    indexer = StringIndexer(inputCol="CATEGORY", outputCol="categoryIndex")
    feature_data = indexer.fit(words_df).transform(words_df)
    cv = CountVectorizer(inputCol="filtered", outputCol="features")
    model = cv.fit(feature_data)
    countVectorizer_feateures = model.transform(feature_data)
    (trainingData, testData) = countVectorizer_feateures.randomSplit([0.8, 0.2],seed = 11)
    nb = NaiveBayes(modelType="multinomial",labelCol="categoryIndex", featuresCol="features")
    nbModel = nb.fit(trainingData)
    nb_predictions = nbModel.transform(testData)
    evaluator = MulticlassClassificationEvaluator(labelCol="categoryIndex", predictionCol="prediction", metricName="accuracy")
    nb_accuracy = evaluator.evaluate(nb_predictions)
    print("Accuracy of NaiveBayes is = %g"% (nb_accuracy))
    print("Test Error of NaiveBayes = %g " % (1.0 - nb_accuracy))
    time2 = datetime.datetime.now()
    elapsedTime = time2 - time1
    print(elapsedTime)