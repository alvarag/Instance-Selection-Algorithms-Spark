package classification.knn

import java.util.logging.Level
import java.util.logging.Logger



import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import classification.abstr.TraitClassifier
import classification.seq.knn.KNNSequential

/**
 * Efficient kNN classifier by using clustering.
 * Implementation of the method presented on "Efficient kNN classification
 * algorithm for big data".
 * Paper: Deng, Z., Zhu, X., Cheng, D., Zong, M., & Zhang, S. (2016).
 * Efficient kNN classification algorithm for big data.
 * Neurocomputing, 195, 143-148.
 *
 * @author Álvar Arnaiz-González
 * @version 2.0
 */
class LCKNN extends TraitClassifier {

  /**
   * Path for logger's messages.
   */
  private val bundleName = "resources.loggerStrings.stringsLCKNN";

  /**
   * Classifier's logger.
   */
  private val logger = Logger.getLogger(this.getClass.getName(), bundleName);

  /**
   * Number of nearest neighbours.
   */
  var k = 1

  /**
   * Set of clusters' centroids.
   */
  var centroidsRDD: RDD[LabeledPoint] = null

  /**
   * Training set with clusters' id.
   */
  var clustersRDD: RDD[(Int, LabeledPoint)] = null

  /**
   * Number of clusters.
   */
  var clustersNum = 10

  /**
   * Number of landmarks.
   */
  var landmarksNum = 100

  /**
   * Landmark selection algorithm: 0 -> random, 1 -> k-means.
   */
  var landmarkSelection = 0

  /**
   * Random seed.
   */
  var seed: Long = 1

  /**
   * Iterations for k-means algorithm.
   */
  var iterations: Int = 10

  /**
   * Bandwidth (h) of a kernel function.
   */
  var bandwidth: Double = 0.4

  /**
   * Nearest neighbours for setting Ẑ elements to non-zero.
   */
  var nonZeroLandmarkNN: Int = 4

  /**
   * Spark's context.
   */
  private var sc: SparkContext = null

  /**
   * kNN calculator (sequential version).
   */
  private var knnSequential: KNNSequential = new KNNSequential(k)

  override def train(trainingSet: RDD[LabeledPoint]): Unit = {
    sc = trainingSet.sparkContext

    // Add an index at the beginning
    var indexedTrainingSet = trainingSet.zipWithIndex.map {
      case (inst, indx) => (indx, inst)
    }

    indexedTrainingSet.cache

    // Create the LSC
    var lsc = new LandmarkSpectralClustering(landmarkSelection, landmarksNum,
      iterations, bandwidth, seed, nonZeroLandmarkNN)

    // Indexed RDD without class
    var indFeaturesTrain = indexedTrainingSet.map {
      case (indx, inst) => (indx, inst.features)
    }

    indFeaturesTrain.cache

    // Compute the matrix Ẑ
    var zHat = lsc.landmarkSpectralClustering(indFeaturesTrain)

    // Create a matrix with Ẑ'
    var zHatMatrix = new IndexedRowMatrix(matrixToIndexedRDD(zHat.transpose))

    // Compute SVD
    var svd = zHatMatrix.computeSVD(clustersNum, true)

    svd.U.rows.cache

    var rowsToTrain = svd.U.toRowMatrix.rows

    // Data is cached for improving efficiency.
    rowsToTrain.cache

    // Compute final kMeans.
    var kMeans = new KMeans().setK(clustersNum).setSeed(seed).setMaxIterations(iterations)
    var kMeansModel = kMeans.run(rowsToTrain)

    // Predict the cluster for each instance
    var clustersPredicted = svd.U.rows.map {
      row => (row.index, kMeansModel.predict(row.vector))
    }

    // Join the clusters with the original data
    var joinTrainClust = indexedTrainingSet.join(clustersPredicted)

    // Assign each instance to each cluster
    clustersRDD = joinTrainClust.map {
      case (indx, inst) => (inst._2, inst._1)
    }

    // Compute the sum of the instances of each cluster
    var aggregatedCentroids = clustersRDD.map({
      case (cluster, inst) => (cluster, inst.features.toArray)
    }).aggregateByKey((Array.ofDim[Double](trainingSet.first().features.size), 0))(
      ((a, b) => (a._1.zip(b).map { case (x, y) => x + y }, a._2 + 1)),
      ((a, b) => (a._1.zip(b._1).map { case (x, y) => x + y }, a._2 + b._2)))

    // Compute the centroids by dividing into the sum
    centroidsRDD = aggregatedCentroids.map {
      case (idCluster, centroids) => (LabeledPoint(idCluster,
        Vectors.dense(centroids._1.map(_ / centroids._2))))
    }
  }

  override def classify(testInstances: RDD[(Long, Vector)]): RDD[(Long, Double)] = {
    // List with clusters
    var clustersList = centroidsRDD.collect().toList

    // Find out the nearest centroid of each instance
    var testNNCentroidsRDD = testInstances.map({
      inst => (inst, clustersList)
    }).map(knnSequential.mapClassify)
    
    // RDD with the index of the instances to test and their NN cluster
    var clusterAssignedTestRDD = testNNCentroidsRDD.join(testInstances).map {
      case (indx, (cluster, inst)) => (cluster.toInt, indx)
    }

    // Join the test instances with the instances of the clusters 
    var testInstancesClustersRDD = clusterAssignedTestRDD.join(clustersRDD)

    // Group by key (index of test instance)
    var grouppedInstancesClustersRDD = testInstancesClustersRDD.map({
      case (cluster, (instIndx, nn)) => (instIndx, nn)
    }).groupByKey().map {
      case (instIndx, nn) => (instIndx, nn.toList)
    }
    
    var instancesTestRDD = grouppedInstancesClustersRDD.join(testInstances).map {
      case (instIndx, (nn, inst)) => ((instIndx, inst), nn)
    }

    // Compute the nearest neighbours
    instancesTestRDD.map(knnSequential.mapClassify)
  }

  override def setParameters(args: Array[String]): Unit = {
    // Check whether or not the number of arguments is even
    if (args.size % 2 != 0) {
      logger.log(Level.SEVERE, "LCKNNPairNumberParamError",
        this.getClass.getName)
      throw new IllegalArgumentException()
    }

    for { i <- 0 until args.size by 2 } {
      try {
        val identifier = args(i)
        val value = args(i + 1)
        assignValToParam(identifier, value)
      } catch {
        case ex: NumberFormatException =>
          logger.log(Level.SEVERE, "LCKNNNoNumberError", args(i + 1))
          throw new IllegalArgumentException()
      }
    }

    // Error if variables has not set correctly
    if (k <= 0 || clustersNum <= 0 || landmarkSelection < 0
      || landmarkSelection > 1 || iterations <= 0 || bandwidth <= 0
      || nonZeroLandmarkNN <= 0 || landmarksNum <= 0) {
      logger.log(Level.SEVERE, "LCKNNWrongArgsValuesError")
      logger.log(Level.SEVERE, "LCKNNPossibleArgs")
      throw new IllegalArgumentException()
    }
  }

  protected def assignValToParam(identifier: String, value: String): Unit = {
    identifier match {
      case "-k" => k = value.toInt
      case "-c" => clustersNum = value.toInt
      case "-t" => landmarkSelection = value.toInt
      case "-l" => landmarksNum = value.toInt
      case "-i" => iterations = value.toInt
      case "-h" => bandwidth = value.toDouble
      case "-z" => nonZeroLandmarkNN = value.toInt
      case "-s" => seed = value.toLong
      case somethingElse: Any =>
        logger.log(Level.SEVERE, "LCKNNWrongArgsError", somethingElse.toString())
        logger.log(Level.SEVERE, "LCKNNPossibleArgs")
        throw new IllegalArgumentException()
    }
  }

  /**
   * Converts a matrix to a RDD of IndexedRows.
   *
   * @param matrix Spark matrix to convert.
   * @return RDD with the matrix in order.
   */
  def matrixToIndexedRDD(matrix: Matrix): RDD[IndexedRow] = {
    val vectors = matrix.rowIter.toArray.zipWithIndex.map {
      case (row, indx) => new IndexedRow(indx, new DenseVector(row.toArray))
    }

    sc.parallelize(vectors)
  }

}
