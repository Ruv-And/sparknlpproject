import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import org.apache.spark.ml.Pipeline
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
import com.johnsnowlabs.nlp.annotators.MultiDateMatcher
import com.johnsnowlabs.nlp.annotator.TextMatcher
import com.johnsnowlabs.nlp.util.io.ReadAs
//import org.apache.spark.sql.textFrameWriter
SparkNLP.version()

val text = spark.read.textFile("/mnt/c/Projects/SparkNLP/NED_vs_POR.txt").toDF("text")

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("bert")

// Then NER can be extracted
val nerTagger = erDLModel.pretrained()
  .setInputCols("sentence", "token", "bert")
  .setOutputCol("ner")

val date = new MultiDateMatcher()
  .setInputCols("document")
  .setOutputCol("date")
  .setAnchorDateYear(2024).setAnchorDateMonth(1).setAnchorDateDay(11)
  
val entityExtractor = new TextMatcher()
  .setInputCols("document", "token")
  .setEntities("/mnt/c/Projects/SparkNLP/match_phrases.txt", ReadAs.TEXT)
  .setOutputCol("entity")
  .setCaseSensitive(false)
  .setTokenizer(tokenizer.fit(text))

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerTagger,
  date,
  entityExtractor
))

val transformedPipeLine = pipeline.fit(text).transform(text)
val pipWrite=transformedPipeLine.select(explode(arrays_zip(col("token.result"),col("ner.result"),col("date.result"),col("entity.result"))).alias("cols"))
.select($"cols"("0"),$"cols"("1"),$"cols"("2"),$"cols"("3"))


pipWrite.na.drop().show(10000, false)


//val modDf = transformedPipeLine.select(col("Entities")).toDF("text")
//modDf.write.format("txt").option("header", "true").save("/mnt/c/Projects/SparkNLP/ArticleEntities")
//write.format("csv").option("header", "true").save("/mnt/c/Projects/SparkNLP/ArticleEntities.csv")
