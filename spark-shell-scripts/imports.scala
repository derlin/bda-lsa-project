import org.apache.spark.ml.linalg.{Vector => ml_Vector}
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition => mllib_SingularValueDecomposition, Vector => mllib_Vector, Vectors => mllib_Vectors}
import org.apache.spark.mllib.clustering.{DistributedLDAModel => mllib_DistributedLDAModel, LDA => mllib_LDA}
import org.apache.spark.ml.clustering.{DistributedLDAModel => ml_DistributedLDAModel, LDA => ml_LDA}

import spark.implicits._
import org.apache.spark.rdd.RDD

import bda.lsa.lda
import bda.lsa.svd

