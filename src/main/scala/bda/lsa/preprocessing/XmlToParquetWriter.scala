package bda.lsa.preprocessing

import org.apache.spark.sql.{Dataset, SparkSession}

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


object XmlToParquetWriter extends App{
  // parse arguments:
  // usage: input-path output-path [num partitions]
  // the partitions argument will determine the number of output files
  val partitions = if (args.length > 0) args(0).toInt else -1
  
  val spark = SparkSession.builder().master("local[*]").getOrCreate()
  
  val assembleMatrix = new AssembleDocumentTermMatrix(spark)
  import assembleMatrix._
  
  var docTexts: Dataset[(String, String)] = parseWikipediaDump(bda.lsa.wikidumpPath)
  if(partitions > 0) docTexts = docTexts.coalesce(partitions)

  /* uncomment this for parquet */
  docTexts.write.parquet(bda.lsa.wikidumpParquetPath)

  /* uncomment this for csv
  docTexts.write.
    //format("com.databricks.spark.csv").
    format("com.databricks.spark.csv").
    option("header", "true").
    save(output)
   */
}
