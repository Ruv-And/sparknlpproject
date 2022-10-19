import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val text = spark.createDataFrame(Seq(
(1, "Vijay Manchiraju absolutely hated his experience at Princeton High School"),
(2, "Aruv Dand still does not have a girl friend")	
)).toDF("id", "text")

val pipeline = PretrainedPipeline("clean_stop", lang = "en")
//text.show()
val transformedPipeLine = pipeline.transform(text)

transformedPipeLine.show()