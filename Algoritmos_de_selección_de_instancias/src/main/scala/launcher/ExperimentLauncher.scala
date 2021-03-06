package launcher

import launcher.execution.ISClassExec
import launcher.execution.TraitExec
import launcher.execution.ISClassSeqExec
import launcher.execution.ClassExec
import launcher.execution.ISClassSeqExecTest
import launcher.execution.ClassSeqExec

/**
 * Pone en marcha la ejecución de una labor de minería de datos.
 *
 * Participante en el patrón de diseño "Strategy" en el que actúa con el
 * rol de contexto ("context"). Se relaciona directamente con la clase
 * [[launcher.execution.TraitExec]] para el uso de este patrón.
 *
 * @author Alejandro González Rogel
 * @version 1.0.0
 */
object ExperimentLauncher {

  /**
   * Ejecuta una nueva tarea de minería de datos.
   *
   * @param args Argumentos para la configuración de la ejecución.
   *   El primer argumento define el tipo de ejecución que se realizará y no
   *   pertenece a la configuración delos componentes del experimento.
   */
  @throws(classOf[IllegalArgumentException])
  def main(args: Array[String]): Unit = {

    val experimentType = args.head

    val execution: TraitExec = experimentType match {
      case "ClassExec"       => new ClassExec
      case "ClassSeqExec"       => new ClassSeqExec
      case "ISClassExec"     => new ISClassExec
      case "ISClassSeqExec"  => new ISClassSeqExec
      case "ISClassSeqExecTest" => new ISClassSeqExecTest
      case _ =>
        throw new IllegalArgumentException(experimentType + "is not an " +
          "execution type.")
    }

    execution.launchExecution(args.drop(1))

  }

}
