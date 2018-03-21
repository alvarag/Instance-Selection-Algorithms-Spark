package instanceSelection.lshis

import java.util.Random
import java.util.logging.Level
import java.util.logging.Logger

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import instanceSelection.abstr.TraitIS

/**
 * Edition instance selection algorithm based on LSH (LSH-Edition).
 *
 * LSH gives an extremely fast performance and quite good balance between
 * filtering and accuracy.
 *
 * @constructor Creates a new LSH-E with default parameters.
 *
 * @author Álvar Arnaiz-González
 * @version 1.0
 */
class LSHEditionIS extends TraitIS {

  /**
   * File with log's sentences.
   */
  private val bundleName = "resources.loggerStrings.stringsLSHEditionIS";

  /**
   * Logger of the algorithm.
   */
  private val logger = Logger.getLogger(this.getClass.getName(), bundleName);

  /**
   * Number of and functions.
   */
  var ANDs: Int = 10

  /**
   * Number of repetitions
   */
  var repeat: Int = 4

  /**
   * Buckets' size.
   */
  var width: Double = 1

  /**
   * Radom seed.
   */
  var seed: Long = 1

  override def instSelection(parsedData: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val r = new Random(seed)
    var steps = repeat

    var inputData = parsedData

    do {
      // New and table functions
      val andTable = new ANDsTable(ANDs, parsedData.first().features.size, width, r.nextInt)

      // Compute the hash of each instance
      val keyInstRDD = inputData.map {
        inst => (andTable.hash(inst), inst)
      }

      keyInstRDD.persist

      // Data set grouped by hash code
      val groupedByHashRDD = keyInstRDD.groupByKey

      // Join hashed test data set
      val joinHashRDD = keyInstRDD.join(groupedByHashRDD)

      // Determine whether or not instances should be retained
      val forFilterRDD = joinHashRDD.map {
        case (hash, (inst, list)) =>
          var remove: Boolean = false

          // If the instance is the only one of that class in the bucket
          //if (mapGroupByLbl.filter(_._1 == inst.label).size == 1)
          if (list.filter(_.label == inst.label).size == 1) {
            // Group instances in the bucket by label.
            val mapGroupByLbl = list.groupBy(_.label)
            // If the bucket contains instances of more than one class
            if (mapGroupByLbl.keySet.size > 1) {
              for { instPerLbl <- mapGroupByLbl } {
                // If there are more than one instance in the bucket of the 
                // other class
                if (instPerLbl._2.size > 1)
                  remove = true
              }
            }
          }

          if (remove)
            (inst, false)
          else
            (inst, true)
      }

      // Select final instances.
      inputData = forFilterRDD.filter(_._2 == true).map {
        case (inst, bool) => inst
      }

      steps = steps - 1
    } while (steps > 0)

    inputData
  }

  override def setParameters(args: Array[String]): Unit = {
    if (args.size % 2 != 0) {
      logger.log(Level.SEVERE, "LSHEditionISPairNumberParamError",
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
          logger.log(Level.SEVERE, "LSHEditionISNoNumberError", args(i + 1))
          throw new IllegalArgumentException()
      }
    }

    if (ANDs <= 0 || width <= 0) {
      logger.log(Level.SEVERE, "LSHEditionISWrongArgsValuesError")
      logger.log(Level.SEVERE, "LSHEditionISPossibleArgs")
      throw new IllegalArgumentException()
    }

  } // end readArgs

  protected override def assignValToParam(identifier: String,
                                          value: String): Unit = {
    identifier match {
      case "-and" => ANDs = value.toInt
      case "-r"   => repeat = value.toInt
      case "-w"   => width = value.toDouble
      case "-s"   => seed = value.toInt
      case somethingElse: Any =>
        logger.log(Level.SEVERE, "LSHEditionISWrongArgsError", somethingElse.toString())
        logger.log(Level.SEVERE, "LSHEditionISPossibleArgs")
        throw new IllegalArgumentException()
    }
  }

}
