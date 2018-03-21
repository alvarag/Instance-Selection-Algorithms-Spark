package instanceSelection.noise

import java.util.logging.Level
import java.util.logging.Logger

import org.apache.spark.mllib.feature.ENN_BD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import instanceSelection.abstr.TraitIS

/**
 * Wrapper for ENN-BD filter algorithm.
 * Available on github: https://github.com/djgarcia/NoiseFramework
 *
 * García-Gil, D., Luengo, J., García, S., & Herrera, F. (2017).
 * Enabling Smart Data: Noise filtering in Big Data classification.
 * arXiv preprint arXiv:1704.01770.
 *
 * @author Álvar Arnaiz-González
 * @version 1.1.0
 */
class ENN extends TraitIS {

  /**
   * File with log's sentences.
   */
  private val bundleName = "resources.loggerStrings.stringsENN";

  /**
   * Logger of the algorithm.
   */
  private val logger = Logger.getLogger(this.getClass.getName(), bundleName);

  override def instSelection(parsedData: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    // Data must be cached in order to improve the performance.
    parsedData.persist

    // Compute the number of classes
    val nClasses = parsedData.map(_.label).distinct().collect().length

    // Create the ENN-BD algorithm.
    val enn_bd_model = new ENN_BD(parsedData, nClasses.toInt, parsedData.first.features.size)

    // Run the filter.
    enn_bd_model.runFilter()
  }

  override def setParameters(args: Array[String]): Unit = {
    if (args.size != 0) {
      logger.log(Level.SEVERE, "ENNWrongArgsValuesError",
        this.getClass.getName)
      throw new IllegalArgumentException()
    }
  }

  protected override def assignValToParam(identifier: String,
                                          value: String): Unit = {
  }

}
