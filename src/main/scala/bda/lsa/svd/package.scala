package bda.lsa

import org.apache.spark.mllib.linalg.{Vector => MLLIBVector}
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.rdd.RDD

/**
  * date: 19.05.17
  *
  * @author Lucy Linder <lucy.derlin@gmail.com>
  */
package object svd {

  /**
    *
    * @param docTermMatrix the matrix docs X tfidfvec as a Dataframe
    * @param k             the number of topics to extract
    * @param featuresCol   the alternative name of the tfidfvec column in the docTermMatrix dataframe
    * @return mat = the RowMatrix used for the model, svd = the svd model
    */
  def runSVD(docTermMatrix: DataFrame, k: Int, featuresCol: String = "tfidfVec"): (RowMatrix, SingularValueDecomposition[RowMatrix, Matrix]) = {

    val vecRdd = bda.lsa.docTermMatrixToRDD(docTermMatrix, featuresCol)
    vecRdd.cache()
    
    val mat = new RowMatrix(vecRdd)
    val svd = mat.computeSVD(k, computeU = true)
    //val u = svd.U.rows.zipWithUniqueId()

    (mat, svd)
  }
}
