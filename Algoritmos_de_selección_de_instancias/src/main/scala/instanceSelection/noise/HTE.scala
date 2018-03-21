package instanceSelection.noise

import java.util.Random
import java.util.logging.Level
import java.util.logging.Logger

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import instanceSelection.abstr.TraitIS
import org.apache.spark.mllib.feature.HTE_BD

/**
 * Wrapper for HTE-BD filter algorithm.
 * Available on github: https://github.com/djgarcia/NoiseFramework
 *
 * García-Gil, D., Luengo, J., García, S., & Herrera, F. (2017).
 * Enabling Smart Data: Noise filtering in Big Data classification.
 * arXiv preprint arXiv:1704.01770.
 *
 * @author Álvar Arnaiz-González
 * @version 1.1.0
 */
class HTE extends TraitIS {

  /**
   * File with log's sentences.
   */
  private val bundleName = "resources.loggerStrings.stringsHTE";

  /**
   * Logger of the algorithm.
   */
  private val logger = Logger.getLogger(this.getClass.getName(), bundleName);

  /**
   * Number of trees.
   */
  var nTrees: Int = 100

  /**
   * Maximum depth.
   */
  var maxDepth: Int = 10

  /**
   * Voting strategy: 0 - Majority or 1 - consensus
   */
  var voting: Int = 0

  /**
   * k nearest neighbours.
   */
  var k: Int = 4

  /**
   * Radom seed.
   */
  var seed: Long = 1

  override def instSelection(parsedData: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    // Data must be cached in order to improve the performance.
    parsedData.persist

    // Compute the number of classes
    val nClasses = parsedData.map({
      inst => (inst.label, 1)
    }).reduceByKey(_ + _).count

    // Create the HTE-BD algorithm.
    val HTE_bd_model = new HTE_BD(parsedData, nTrees, k, voting,
      nClasses.toInt, parsedData.first.features.size, maxDepth, seed.toInt)

    // Run the filter.
    HTE_bd_model.runFilter()
  }

  override def setParameters(args: Array[String]): Unit = {
    if (args.size % 2 != 0) {
      logger.log(Level.SEVERE, "HTEPairNumberParamError",
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
          logger.log(Level.SEVERE, "HTENoNumberError", args(i + 1))
          throw new IllegalArgumentException()
      }
    }

    if (nTrees <= 0 || maxDepth <= 0 || k <= 0 || voting < 0 || voting > 1) {
      logger.log(Level.SEVERE, "HTEWrongArgsValuesError")
      logger.log(Level.SEVERE, "HTEPossibleArgs")
      throw new IllegalArgumentException()
    }

  } // end readArgs

  protected override def assignValToParam(identifier: String,
                                          value: String): Unit = {
    identifier match {
      case "-t" => nTrees = value.toInt
      case "-m" => maxDepth = value.toInt
      case "-v" => voting = value.toInt
      case "-k" => k = value.toInt
      case "-s" => seed = value.toInt
      case somethingElse: Any =>
        logger.log(Level.SEVERE, "HTEWrongArgsError", somethingElse.toString())
        logger.log(Level.SEVERE, "HTEPossibleArgs")
        throw new IllegalArgumentException()
    }
  }

}
