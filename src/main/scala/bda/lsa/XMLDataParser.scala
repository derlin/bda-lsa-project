package bda.lsa

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset

// author: Lucy Linder <lucy.derlin@gmail.com>
// date: 15.05.17
// Parse XML wikidump, clean, remove stopwords, lemmatize and finally save the result to HDFS.
//
// ## Launch on the Daplab
// With this configuration, it took 21m32s
// ```
// export SPARK_MAJOR_VERSION=2
// export HADOOP_CONF_DIR=/etc/hadoop/conf
// export LD_LIBRARY_PATH=/usr/hdp/current/hadoop-client/lib/native:$LD_LIBRARY_PATH
// spark-submit --class bda.lsa.XMLDataParser --master yarn-client \
//    --num-executors 8 --driver-memory 10G --executor-memory 20g \
//    bda-prepare-data.jar /shared/wikipedia/wikidump.xml /shared/wikipedia/wikidump-parquet 20
// ```
// ## Reloading data
// to reload CSV data, use:
// ```
// sqlContext.read.format("com.databricks.spark.csv").option("header", true).option("inferSchema", true).load("/path/to/directory/*")
// ```
// to reload other formats, use:
// ```
// sqlContext.read.format("parquet").load("/path/to/directory/*")
// ```
//


object XMLDataParser extends App{
  // parse arguments:
  // usage: input-path output-path [num partitions]
  // the partitions argument will determine the number of output files
  val path = if(args.length > 0) args(0) else "/shared/wikipedia/wikidump.xml"
  val output = if(args.length > 1) args(1) else "/shared/wikipedia/wikidump-csv"
  val partitions = if (args.length > 2) args(2).toInt else -1

  val stopwordsFile = "src/main/resources/stopwords.txt"

  val spark = SparkSession.builder().getOrCreate()
  
  val assembleMatrix = new AssembleDocumentTermMatrix(spark)
  import assembleMatrix._
  
  var docTexts: Dataset[(String, String)] = parseWikipediaDump(path)
  if(partitions > 0) docTexts = docTexts.coalesce(partitions)

  /* uncomment this for parquet */
  docTexts.write.parquet(output)

  /* uncomment this for csv
  docTexts.write.
    //format("com.databricks.spark.csv").
    format("com.databricks.spark.csv").
    option("header", "true").
    save(output)
   */
}
