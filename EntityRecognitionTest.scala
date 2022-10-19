import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val text = spark.createDataFrame(Seq(
(1, "Vijay Manchiraju absolutely hated his experience at Princeton High School"),
(2, "Aruv Dand still does not have a girl friend"),
(3, "My address is 19 Ridgeview dr, Elizabeth, New Jersey"),
(4, "My Email is vmanchiraju800@gmail.com")	
)).toDF("id", "text")

val pipeline = PretrainedPipeline("recognize_entities_dl", lang = "en")
val transformedPipeLine = pipeline.transform(text)

transformedPipeLine.show()

transformedPipeLine.select("entities.result").show(false)