/******************************************************************************
 *                                                                            *
 * Copyright (C) 2018 Fondazione Istituto Italiano di Tecnologia (IIT)        *
 * All Rights Reserved.                                                       *
 *                                                                            *
 ******************************************************************************/

/**
 * @file Metric.h
 * @authors: Valentina Vasco <valentina.vasco@iit.it>
 */

#ifndef __METRIC_H__
#define __METRIC_H__

#include <iostream>
#include <map>

#include <yarp/os/all.h>
#include <yarp/sig/Vector.h>

#include <AssistiveRehab/skeleton.h>

using namespace std;
using namespace yarp::sig;
using namespace assistive_rehab;

class Metric
{

public:
    Metric();
    virtual ~Metric();
    virtual void print();
    virtual string getName() const = 0;
    virtual string getTagJoint() const = 0;
    virtual string getMotionType() const = 0;
    virtual Vector getRefDir() const = 0;
    virtual string getTagPlane() const = 0;
    virtual double getRangePlane() const = 0;
    virtual double getMax() const = 0;
    virtual double getMin() const = 0;
    virtual double getDuration() const = 0;
    virtual map<string, pair<string,double>> getInitialConf() const = 0;
};

class Rom : public Metric
{
    string name;
    string motion_type;
    string tag_joint;
    Vector ref_dir;
    string tag_plane;
    double range_plane;
    double min;
    double max;
    double duration;
    map<string, pair<string,double>> keypoints2conf;

public:
    Rom();
    Rom(const string &name_, const string &motion_type_, const string &tag_joint_, const Vector &ref_dir_,
        const string &tag_plane_, const double &range_plane_, const double &min_, const double &max_, const double &duration_,
        const map<string, pair<string,double>> &keypoints2conf_);

    string getName() const { return name; }
    string getTagJoint() const { return tag_joint; }
    string getMotionType() const { return motion_type; }
    Vector getRefDir() const { return ref_dir; }
    string getTagPlane() const { return tag_plane; }
    double getRangePlane() const { return range_plane; }
    double getMax() const { return max; }
    double getMin() const { return min; }
    double getDuration() const { return duration; }
    map<string, pair<string,double>> getInitialConf() const { return keypoints2conf; }
    void print();

};


#endif
