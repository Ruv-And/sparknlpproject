import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.DataFrame

Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");

StopWordsRemover remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered");

DataFrame jsondf = sqlContext.read().json("C:/Projects/SparkNLP/articles.json");

DataFrame wordsDataFrame = tokenizer.transform(jsondf);

DataFrame filteredTokens = remover.transform(wordsDataFrame);
filteredTokens.show();

CountVectorizerModel cvModel = new CountVectorizer()
        .setInputCol("filtered").setOutputCol("features")
        .setVocabSize(10).fit(filteredTokens);
cvModel.transform(filteredTokens).show();