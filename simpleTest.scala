import java.io.PrintWriter
//read json file into dataframe
val df = spark.read.option("multiline","true").json("/home/uadmin/employee.json")
df.printSchema()
df.show(false)
val filePath = "/mnt/c/Projects/SparkNLP/employees3.csv"
//new PrintWriter(filePath) { write(df.select(col("firstname").alias("fname"),col("lastname"))); close }
val modDf = df.select(col("firstname").alias("fname"),col("lastname"))
modDf.write.format("csv").option("header", "true").save(filePath)

