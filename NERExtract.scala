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
import org.apache.spark.sql.DataFrameWriter

val data = Seq("This agreement signed between Ensono Inc and RexWare Inc. with an Agreement Date 10/01/2020 with a Contract Start Date 12/01/2020 and Contract End Date 12-01-2021. COLA Adjustment will be 3% per year starting on 01/01/2023").toDF("text")

// First extract the prerequisites for the NerDLModel
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
val nerTagger = NerDLModel.pretrained()
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
  .setTokenizer(tokenizer.fit(data))

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerTagger,
  date,
  entityExtractor
))


val result = pipeline.fit(data).transform(data)
//result.printSchema()
//result.show()
//result.select("token.result","ner.result","date.result","entity.result").show(3, truncate=50)

//val resultDates=result.selectExpr("explode(date.result)").show()


val dfWrite=result.select(explode(arrays_zip(col("token.result"),col("ner.result"),col("date.result"),col("entity.result"))).alias("cols"))
.select($"cols"("0"),$"cols"("1"),$"cols"("2"),$"cols"("3"))
       // expr(cols("1")).alias("pos"),
       // expr(cols("2")).alias("ner_label"),expr(cols("3")).alias("date_name")).show()
//element_at($cols, $0)
dfWrite.show(1000)
dfWrite.write.option("header",true)
   .csv("/mnt/c/Projects/SparkNLP/OutNERExtr.txt")

//val testData = spark.createDataFrame(Seq(
//(1, "Google has announced the release of a beta version of the 19/01/2021 popular TensorFlow machine learning library"),
//(2, "The Paris metro will soon enter the 21st century, ditching single-use paper tickets for rechargeable electronic cards.")
//)).toDF("id", "text")

//val ptpipeline = PretrainedPipeline("explain_document_ml", lang="en")

//val annotation = ptpipeline.transform(testData)

//annotation.show()


