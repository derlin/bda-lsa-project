
import org.apache.spark.ml.linalg.{Vector => ml_Vector}
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition => mllib_SingularValueDecomposition, Vector => mllib_Vector, Vectors => mllib_Vectors}
import org.apache.spark.mllib.clustering.{DistributedLDAModel => mllib_DistributedLDAModel, LDA => mllib_LDA}
import org.apache.spark.ml.clustering.{DistributedLDAModel => ml_DistributedLDAModel, LDA => ml_LDA}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}

import bda.lsa._
import bda.lsa.lda
import bda.lsa.svd

val data = getData(spark)
val svdModel = svd.RunSVD.loadModel(spark)
val qs = new svd.SVDQueryEngine(svdModel, data)

val ldaModel = lda.ml.RunLDA.loadModel(spark)
val ql = new lda.ml.LDAQueryEngine(ldaModel,data)

val docIds = data.docIds.collectAsMap
val topics = ql.describeTopicsWithWords(5)


def qlTopTopicsForTerm(s: String) =
  ql.topTopicsForTerm(data.termIds.indexOf(s)).map(t => s"${t._1}, ${topics(t._1)}, ${t._2}").mkString("\n")


def qlTopTopicsForDocument(id : Int) =
  ql.topTopicsForDocument(id).map(t => s"${t._1}, ${topics(t._1)}, ${t._2}").mkString("\n")



