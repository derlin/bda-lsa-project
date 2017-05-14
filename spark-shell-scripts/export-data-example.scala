/**
  * date: 14.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */

// this exports what "load-data-example.scala" outputs

docTermMatrix.write.format("parquet").save("/tmp/docTermMatrix.parquet")
val dtm = spark.sqlContext.read.parquet("/tmp/docTermMatrix.parquet")

sc.parallelize(termIds).saveAsTextFile("/tmp/termIds")
val tids = sc.textFile("/tmp/termIds").collect.toArray

sc.parallelize(docIds.toSeq).toDF.coalesce(1).write.format("json").save("/tmp/docIds")
val dids = spark.read.json("/tmp/docIds").map {
  r => (r.getAs[Long](0), r.getAs[String](1))
}.collect.toMap