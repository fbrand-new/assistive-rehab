/******************************************************************************
 *                                                                            *
 * Copyright (C) 2019 Fondazione Istituto Italiano di Tecnologia (IIT)        *
 * All Rights Reserved.                                                       *
 *                                                                            *
 ******************************************************************************/

/**
 * @file idl.thrift
 * @authors: Valentina Vasco <valentina.vasco@iit.it>
 */

struct Matrix { }
(
   yarp.name="yarp::sig::Matrix"
   yarp.includefile="yarp/sig/Matrix.h"
)

struct Property { }
(
   yarp.name="yarp::os::Property"
   yarp.includefile="yarp/os/Property.h"
)

/**
 * motionAnalyzer_IDL
 *
 * IDL Interface to Motion Analyzer services.
 */
service motionAnalyzer_IDL
{
   /**
   * Load exercise to analyze.
   * @param exercise_tag name of the exercise to analyze
   * @return true/false on failure.
   */
   bool loadExercise(1:string exercise_tag);

   /**
   * Get the name of the exercise begin performed.
   * @return string containing the name of the exercise begin performed / empty string on failure.
   */
   string getExercise();

   /**
   * List available exercises.
   * @return the list of the available exercises as defined in the motion-repertoire.
   */
   list<string> listExercises();

   /**
   * Start processing.
   * @param use_robot_template true if robot template is used.
   * @return true/false on success/failure.
   */
   bool start(1:bool use_robot_template);

   /**
   * Stop feedback.
   * @return true/false on success/failure.
   */
   bool stopFeedback();

   /**
   * Stop processing.
   * @return true/false on success/failure.
   */
   bool stop();

   /**
   * Select skeleton by its tag.
   * @param skel_tag tag of the skeleton to process
   * @return true/false on success/failure.
   */
   bool selectSkel(1:string skel_tag);

   /**
   * List joints on which feedback is computed.
   * @return the list of joints on which feedback is computed.
   */
   list<string> listJoints()

   /**
   * Select property to visualize.
   * @param prop property to visualize
   * @return true/false on success/failure.
   */
   bool selectMetricProp(1:string prop_tag);

   /**
   * List the available properties computable for the current metric.
   * @return the list of the available properties computable for the current metric.
   */
   list<string> listMetricProps()

   /**
   * Select metric to analyze.
   * @param metric_tag metric to analyze
   * @return true/false on success/failure.
   */
   bool selectMetric(1:string metric_tag);

   /**
   * List the available metrics for the current exercise.
   * @return the list of the available metrics for the current exercise.
   */
   list<string> listMetrics()

   /**
   * Get the metric to visualise.
   * @return metric to visualise.
   */
   string getCurrMetricProp();

   /**
   * Select the part to move.
   * @return true/false on success/failure.
   */
   bool setPart(1:string part);

   /**
   * Select the template for the analysis.
   * @return true/false on success/failure.
   */
   bool setTemplateTag(1:string template_tag);

   /**
   * Mirror the skeleton template.
   * @param robot_skeleton_mirror if true, robot template has to be mirrored.
   * @return true/false on success/failure.
   */
   bool mirrorTemplate(1:bool robot_skeleton_mirror);

   /**
   * Set the pose of the finish line.
   * @param pose of the finish line.
   * @return true/false on success/failure.
   */
   bool setStartLinePose(1:list<double> start_line_pose);
   bool setFinishLinePose(1:list<double> finish_line_pose);
   
   /**
   * Retrieve motion analysis result.
   * @return a property-like object containing the result of the analysis.
   */
   Property getState();
}
