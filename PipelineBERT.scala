import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import com.johnsnowlabs.nlp.training._

val trainingData = CoNLL().readDataset(spark, "/mnt/c/Projects/SparkNLP/dataSet.txt")
trainingData.selectExpr("text", "token.result as tokens", "label.result as label")
  .show(false)

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained(trainingData)
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  tokenClassifier
))

val data = Seq("Gunter Lexare was born in Mannheim and lived in Paris. Harrison James is terrible at FIFA").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("label.result").show(false)