/******************************************************************************
 *                                                                            *
 * Copyright (C) 2018 Fondazione Istituto Italiano di Tecnologia (IIT)        *
 * All Rights Reserved.                                                       *
 *                                                                            *
 ******************************************************************************/

/**
 * @file Processor.h
 * @authors: Valentina Vasco <valentina.vasco@iit.it>
 */

#ifndef __PROCESSOR_H__
#define __PROCESSOR_H__

#include <iostream>
#include <map>
#include <math.h>

#include <yarp/sig/Matrix.h>
#include <yarp/math/Math.h>

#include <AssistiveRehab/skeleton.h>
#include "Metric.h"

using namespace std;
using namespace yarp::sig;
using namespace assistive_rehab;

class Processor;

Processor* createProcessor(const string& motion_tag, const Metric* metric_);

class Processor
{

protected:
    SkeletonStd skeleton_init;
    map<string, string> keypoints2conf;
    SkeletonStd curr_skeleton;
    const Metric* metric;

public:
    Processor();
    virtual ~Processor() {;}
//    virtual string getMotionType();
    virtual void setInitialConf(const SkeletonStd& skeleton_init_, const map<string, string>& keypoints2conf_) = 0;
    bool isStatic(const KeyPoint& keypoint);
    void update(const SkeletonStd &curr_skeleton_);
    bool isDeviatingFromIntialPose();
    bool isDeviatingFromIntialPose(const KeyPoint &keypoint, const KeyPoint &keypoint_init);
    virtual double computeMetric() {;}
    virtual string getProcessedMetric() = 0;

};

class Rom_Processor : public Processor
{
    const Rom* rom;

public:

    Rom_Processor();
    Rom_Processor(const Metric *rom_);
    void setInitialConf(const SkeletonStd& skeleton_init_, const map<string, string>& keypoints2conf_) override;
    double computeMetric();
    string getProcessedMetric() { return rom->getName(); }

    static const string motion_type;


};

#endif
