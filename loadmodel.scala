import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.LightPipeline
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.{StopWordsCleaner, Tokenizer}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, explode, size}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach



import com.johnsnowlabs.nlp.embeddings.{WordEmbeddingsModel,BertEmbeddings}
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{Benchmark, FileHelper}
import scala.io.Source
   
    
	  
	val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
	
	val sentence =new SentenceDetector().setInputCols("document").setOutputCol("sentence")

    val token = new Tokenizer().setInputCols("sentence").setOutputCol("token")
	val trainingData = CoNLL().readDataset(spark, "/mnt/c/Projects/SparkNLP/dataSet.txt")
    trainingData.selectExpr("text", "token.result as tokens", "label.result as label").show(false)
	
	val bert2 = BertEmbeddings.pretrained("small_bert_L2_768", "en").setInputCols("sentence","token").setOutputCol("bert").setCaseSensitive(false)
    
	println("Hello, world! bert2") 
	
	val loaded_ner_model = NerDLModel.read.load("./NER_bert_prod_20200221")
	.setInputCols("sentence", "token", "bert")
    .setOutputCol("ner")
	
	println("Hello, world!more ") 
	val converter=new NerConverter()  
	  .setInputCols("document", "token", "ner")
	  .setOutputCol("ner_span")
	  
	//val chunker = new NerChunker()
    //.setInputCols(Array("sentence", "ner"))
    //.setOutputCol("ner_chunk")
   //.setRegexParsers(Array("<ORG>.*<MODEL>"))
	  
	 println("Hello, world!more 3 ")  
	  
	  val custom_ner_pipeline = new Pipeline()
      .setStages(Array(
        document,sentence,token,bert2,
        loaded_ner_model,converter
      ))
	  
println("Hello, world!more 4 ")  
val text = Seq("Panasonic KXP-1105", "Toshiba TRV-45","U.N. official Ekeus heads for Baghdad")

val prediction_data = text.toDF("text")
val prediction_model = custom_ner_pipeline.fit(prediction_data)
println("Hello, world!more 5 ")  
val preds = prediction_model.transform(prediction_data)
preds.printSchema()
//preds.show(truncate = false) 
println("Hello, world!more 6 ") 
//val predsShow=preds.select("token.result","bert.result","ner.result","ner.annotatorType","ner.embeddings","label.result").show(3, truncate=50)
val predsShow=preds.select("token.result","bert.result","ner.result","ner.annotatorType","ner.embeddings").show(3, truncate=50)
//predsShow.show(truncate = false) 



//select(explode(arrays_zip('ner_span.result', 'pos.result',  'label.result')).alias("cols")


//    val pipelineDF = pipeline2.fit(training_data).transform(training_data)



  //  pipelineDF.select("ner").show(1, false)








//val explainDocumentPipeline = PretrainedPipeline("explain_document_ml")
//val lightPipeline = new LightPipeline(explainDocumentPipeline.model)
//println(lightPipeline.annotate("Hello world, please annotate my text"))
//val trainingConll = CoNLL().readDataset(spark, "/mnt/c/Projects/SparkNLP/Product_CoNLL.txt")
//val embeddings = BertEmbeddings.pretrained("small_bert_L2_128", "en").setInputCols("document", "cleanTokens").setOutputCol("embeddings").setCaseSensitive(true)

