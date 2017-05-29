package classification.knn

import java.util.logging.Level
import java.util.logging.Logger

import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.KNNClassificationModel
import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import classification.abstr.TraitClassifier

/**
 * Wrapper for kNN classifier by using Hybrid Spill Trees.
 * The original classifier is from saurfang and is available on Github:
 * https://github.com/saurfang/spark-knn
 *
 * @author Álvar Arnaiz-González
 * @version 2.0
 */
class SpillTreeKNN extends TraitClassifier {

  /**
   * Path for logger's messages.
   */
  private val bundleName = "resources.loggerStrings.stringsSpillTreeKNN";

  /**
   * Classifier's logger.
   */
  private val logger = Logger.getLogger(this.getClass.getName(), bundleName);

  /**
   * Number of nearest neighbours.
   */
  var k = 1

  /**
   * Conjunto de datos almacenado tras la etapa de entrenamiento.
   */
  var trainingData: RDD[LabeledPoint] = null

  /**
   * Top tree size.
   */
  var t = 100

  /**
   * Random seed.
   */
  var seed: Long = 1

  var kNNClassModel: KNNClassificationModel = null

  /**
   * Spark's context.
   */
  private var sc: SparkContext = null

  override def train(trainingSet: RDD[LabeledPoint]): Unit = {
    var kNNClassifier: KNNClassifier = null

    // Assign training data.
    trainingData = trainingSet
    trainingData.persist

    sc = trainingSet.sparkContext

    // Initialize the kNN model.
    if (t == 0)
      kNNClassifier = new KNNClassifier().setTopTreeSize(
        trainingSet.count().toInt / 500).setK(k)
    else
      kNNClassifier = new KNNClassifier().setTopTreeSize(t).setK(k)

    // Convert the RDD to Dataframe.
    val sparkSession = SparkSession.builder().getOrCreate()

    var ts = trainingSet.map {
      inst =>
        org.apache.spark.ml.feature.LabeledPoint(inst.label,
          new DenseVector(inst.features.toArray))
    }

    var df = sparkSession.createDataFrame(ts)

    // Train the model.
    kNNClassModel = kNNClassifier.fit(df)
  }

  override def classify(testInstances: RDD[(Long, Vector)]): RDD[(Long, Double)] = {
    val sparkSession = SparkSession.builder().getOrCreate()

    // Convert testInstances RDD to DataFrame
    var ts = testInstances.map {
      case (id, features) =>
        org.apache.spark.ml.feature.LabeledPoint(id,
          new DenseVector(features.toArray))
    }

    var df = sparkSession.createDataFrame(ts)

    var pred = kNNClassModel.transform(df)

    // Compose the solution RDD
    pred.select("label", "prediction").rdd.map {
      row =>
        (row.get(0).toString().toDouble.toLong, row.get(1).toString().toDouble)
    }
  }

  override def setParameters(args: Array[String]): Unit = {
    // Check whether or not the number of arguments is even
    if (args.size % 2 != 0) {
      logger.log(Level.SEVERE, "SpillTreeKNNPairNumberParamError",
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
          logger.log(Level.SEVERE, "SpillTreeKNNNoNumberError", args(i + 1))
          throw new IllegalArgumentException()
      }
    }

    // Error if variables has not set correctly
    if (k <= 0 || t < 0) {
      logger.log(Level.SEVERE, "SpillTreeKNNWrongArgsValuesError")
      logger.log(Level.SEVERE, "SpillTreeKNNPossibleArgs")
      throw new IllegalArgumentException()
    }
  }

  protected def assignValToParam(identifier: String, value: String): Unit = {
    identifier match {
      case "-k" => k = value.toInt
      case "-t" => t = value.toInt
      case "-s" => seed = value.toLong
      case somethingElse: Any =>
        logger.log(Level.SEVERE, "SpillTreeKNNWrongArgsError", somethingElse.toString())
        logger.log(Level.SEVERE, "SpillTreeKNNPossibleArgs")
        throw new IllegalArgumentException()
    }
  }

}
