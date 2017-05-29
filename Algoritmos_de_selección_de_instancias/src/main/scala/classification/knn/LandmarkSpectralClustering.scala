package classification.knn

import scala.collection.mutable.ListBuffer

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.SparseMatrix
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

import classification.seq.knn.KNNSequential

/**
 * Utility class for the LCkNN classifier. It computes the Ẑ matrix.<br/>
 *
 * Based on Julia implementation of LSC:
 * http://int8.io/large-scale-spectral-clustering-with-landmark-based-representation/
 *
 * @author Álvar Arnaiz-González
 * @version 2.0
 */
class LandmarkSpectralClustering(landmarkSelection: Int,
                                 landmarksNum: Int,
                                 iterations: Int,
                                 bandwidth: Double,
                                 seed: Long,
                                 nearestLandmarksNum: Int) extends Serializable {

  /**
   * kNN calculator (sequential version).
   */
  private var knnSequential: KNNSequential = new KNNSequential

  /**
   * Computes Landmark-based Spectral Clustering on the training data.
   *
   * @param  trainingData Training data for clustering.
   * @return Ẑ matrix.
   */
  def landmarkSpectralClustering(trainingData: RDD[(Long, Vector)]) = {
    var landmarkPoints: Array[Vector] = null

    // Random sampling
    if (landmarkSelection == 0) {
      landmarkPoints = randomSample(trainingData)
    } // k-means sampling
    else {
      landmarkPoints = kMeansSample(trainingData)
    }

    // Compute the Ẑ matrix
    composeZHatMatrix(trainingData, landmarkPoints)
  }

  /**
   * Computes the Ẑ matrix from the training set.
   *
   * @param trainingData Training data for building Ẑ matrix.
   * @return Ẑ matrix.
   */
  private def composeZHatMatrix(trainingData: RDD[(Long, Vector)],
                                landmarkPoints: Array[Vector]): DenseMatrix = {
    var numInsts = trainingData.count().toInt
    var sum = 0.0

    // Compute the distance between training instances and landmarks.
    var distances = trainingData.map({
      instance => (instance, landmarkPoints)
    }).map(knnSequential.distance)

    // Compute similarities between instances according to the Gaussian kernel.
    var similarities = distances.map {
      case (indx, inst, distances) => (indx, distances.map(gaussianKernel).toVector)
    }

    // Content of the sparse matrix with similarities
    var zHatContent = similarities.flatMap(closestLandmarks)

    // Sparse matrix Ẑ
    var zHat = SparseMatrix.fromCOO(numInsts, landmarksNum, zHatContent.collect())

    // Compute the row sum of W.
    var d = ListBuffer.empty[(Int, Int, Double)]

    // Ẑ'
    var zHatT = zHat.transpose

    var i = 0
    for (row <- zHatT.rowIter) {
      sum = row.toArray.sum

      // D^(-1/2) 
      d += Tuple3(i, i, math.pow(sum, -0.5))

      i += 1
    }

    // Create the diagonal sparse matrix with d
    var diagonal = SparseMatrix.fromCOO(landmarksNum, landmarksNum, d)

    // Return the multiplication: D^(-1/2)·Z
    diagonal.multiply(zHat.transpose.toDense)
  }

  /**
   * Computes the closest landmarks of each instance.
   *
   * @param tuple Tuple composed of (id, distances).
   * @return List of matrix' coordinates: (i, j, distance).
   */
  private def closestLandmarks(tuple: Tuple2[Long, scala.Vector[Double]]): List[(Int, Int, Double)] = {
    var indx = tuple._1
    var similarities = tuple._2

    // Find the closest r landmarks for each instance
    var indexes = similarities.zipWithIndex.
      sortBy(_._1).reverse.take(nearestLandmarksNum)

    // Computes the sum of the closest landmarks
    var sum = indexes.map({ case (dist, idx) => dist }).sum

    var tuples = indexes.map {
      case (dist, idx) => (indx.toInt, idx.toInt, dist / sum)
    }

    tuples.toList
  }

  /**
   * Returns a set of centroids randomly selected without replacement.
   *
   * @param  trainingData Training data set.
   * @return random centroids.
   */
  private def randomSample(trainingData: RDD[(Long, Vector)]): Array[Vector] = {
    var fraction = landmarksNum.toDouble / trainingData.count()

    trainingData.sample(false, fraction, seed).take(landmarksNum).map {
      case (indx, inst) => inst
    }
  }

  /**
   * Returns the centroids of the clusters computed by k-means algorithm.
   *
   * @param  trainingData Training data set.
   * @return centroids of the clusters.
   */
  private def kMeansSample(trainingData: RDD[(Long, Vector)]): Array[Vector] = {
    var kMeans = new KMeans().setK(landmarksNum).setSeed(seed).
      setMaxIterations(iterations)

    var train = trainingData.map {
      case (indx, inst) => inst
    }

    kMeans.run(train).clusterCenters
  }

  /**
   * Computes the Gaussian kernel on distances.
   * The kernel used is exp(-distance/2h²).
   *
   * @param distance Distance.
   * @return Gaussian kernel.
   */
  private def gaussianKernel(distance: Double): Double = {
    math.exp(-distance / (2 * math.pow(bandwidth, 2.0)))
  }

}