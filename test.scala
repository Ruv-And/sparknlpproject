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


import com.johnsnowlabs.nlp.embeddings.{WordEmbeddingsModel}



import com.johnsnowlabs.nlp.training.CoNLL



import com.johnsnowlabs.nlp.util.io.ResourceHelper







import com.johnsnowlabs.util.{Benchmark, FileHelper}















import scala.io.Source




val documentAssembler = new DocumentAssembler().

    setInputCol("text").

    setOutputCol("document")



val regexTokenizer = new Tokenizer().

    setInputCols(Array("sentence")).

    setOutputCol("token")

val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")



val finisher = new Finisher()

    .setInputCols("token")

    .setIncludeMetadata(true)



val pipeline = new Pipeline().

    setStages(Array(

        documentAssembler,

        sentenceDetector,

        regexTokenizer,

        finisher

    ))

val data1 = Seq("hello, this is an example sentence").toDF("text")
val prediction = pipeline.fit(data1).transform(data1)
prediction.show()



val conll = CoNLL(explodeSentences = false)



    val training_data = conll.readDataset(ResourceHelper.spark, "/mnt/c/Projects/SparkNLP/PROD_ECOMM.txt")
	//val training_data = conll.readDataset(ResourceHelper.spark, "/mnt/c/Projects/SparkNLP/Product_CoNLL.txt")
	

    //training_data.show()

    val bert = BertEmbeddings.pretrained("small_bert_L2_768", "en").setInputCols("sentence","token").setOutputCol("bert").setCaseSensitive(false)
    val nerTagger = new NerDLApproach().setInputCols("sentence", "token", "bert").setLabelColumn("label").setOutputCol("ner")
	.setEvaluationLogExtended(true)
	.setEnableOutputLogs(true)
	.setIncludeConfidence(true)


val ner_pipeline = new Pipeline().setStages(Array(
  bert,
  nerTagger
))
val ner_model = ner_pipeline.fit(training_data)

ner_model.stages.last.asInstanceOf[NerDLModel].write.overwrite().save("NER_bert_prod_20200221")

//ner_model.write.overwrite().save("NER_bert_prod_20200221")



    
	  
	val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
	
	val sentence =new SentenceDetector().setInputCols("document").setOutputCol("sentence")

    val token = new Tokenizer().setInputCols("sentence").setOutputCol("token")
	
	val bert2 = BertEmbeddings.pretrained("small_bert_L2_768", "en").setInputCols("sentence","token").setOutputCol("bert").setCaseSensitive(false)
    
	println("Hello, world! bert2") 
	
	val loaded_ner_model = NerDLModel.load("./NER_bert_prod_20200221")
   .setInputCols("sentence", "token", "bert")
   .setOutputCol("ner")
	
	println("Hello, world!more ") 
	val converter=new NerConverter()  
	  .setInputCols("document","token", "ner")
	  .setOutputCol("ner_span")
	  
	  	println("Hello, world!more 2") 
	  
	  val custom_ner_pipeline = new Pipeline()
      .setStages(Array(
        document,sentence,token,bert,
        loaded_ner_model,converter
      ))
	  

val text = Seq("This is my first sentence. MH01-M-Black  Pansonic KXP-1100 This my second. Chaz Kangeroo Hoodie-M-Black")

val prediction_data = text.toDF("text")
val prediction_model = custom_ner_pipeline.fit(prediction_data)
val preds = prediction_model.transform(prediction_data)
preds.show()





//    val pipelineDF = pipeline2.fit(training_data).transform(training_data)



  //  pipelineDF.select("ner").show(1, false)








//val explainDocumentPipeline = PretrainedPipeline("explain_document_ml")
//val lightPipeline = new LightPipeline(explainDocumentPipeline.model)
//println(lightPipeline.annotate("Hello world, please annotate my text"))
//val trainingConll = CoNLL().readDataset(spark, "/mnt/c/Projects/SparkNLP/Product_CoNLL.txt")
//val embeddings = BertEmbeddings.pretrained("small_bert_L2_128", "en").setInputCols("document", "cleanTokens").setOutputCol("embeddings").setCaseSensitive(true)

