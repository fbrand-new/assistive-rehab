/*
 * Copyright (C) 2016 iCub Facility - Istituto Italiano di Tecnologia
 * Author: Valentina Vasco
 * email:  valentina.vasco@iit.it
 * Permission is granted to copy, distribute, and/or modify this program
 * under the terms of the GNU General Public License, version 2 or any
 * later version published by the Free Software Foundation.
 *
 * A copy of the license can be found at
 * http://www.robotcub.org/icub/license/gpl.txt
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details
*/

#include <yarp/os/all.h>
#include <yarp/sig/Vector.h>

#include "motionAnalyzer_IDL.h"

using namespace std;
using namespace yarp::os;
using namespace yarp::sig;


/********************************************************/
class Manager : public RFModule,
                public motionAnalyzer_IDL
{

    RpcClient opcPort;
    RpcServer rpcPort;

    ResourceFinder *rf;

    struct listEntry
    {
        string tag;
        int id_joint;
        int min;
        int max;
    };
    map<string, listEntry> motion_list;

    Vector j_3d;

    /********************************************************/
    bool load()
    {
        ResourceFinder rf;
        rf.setVerbose();
        rf.setDefaultContext(this->rf->getContext().c_str());
        rf.setDefaultConfigFile(this->rf->find("configuration-file").asString().c_str());
        rf.configure(0, NULL);

        Bottle &bGeneral = rf.findGroup("GENERAL");

        if(!bGeneral.isNull())
        {
            int nmovements = bGeneral.find("number_movements").asInt();

            if(Bottle *motion_tag = bGeneral.find("motion_tag").asList())
            {
                if(Bottle *n_motion_tag = bGeneral.find("number_motion").asList())
                {
                    for(int i=0; i<motion_tag->size(); i++)
                    {
                        string curr_tag = motion_tag->get(i).asString();
                        int motion_number = n_motion_tag->get(i).asInt();

                        for(int j=0; j<motion_number; j++)
                        {
                            Bottle &bMotion = rf.findGroup(curr_tag+"_"+to_string(j));
                            if(!bMotion.isNull())
                            {
                                if(Bottle *bJoint = bMotion.find("tag").asList())
                                {
                                    motion_list[curr_tag+"_"+to_string(j)].tag = bJoint->get(0).asString();
                                    motion_list[curr_tag+"_"+to_string(j)].id_joint = bJoint->get(1).asInt();
                                    motion_list[curr_tag+"_"+to_string(j)].min = bMotion.check("min",Value(-1)).asInt();
                                    motion_list[curr_tag+"_"+to_string(j)].max = bMotion.check("max",Value(-1)).asInt();
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            yError() << "Error in loading parameters. Stopping module!";
            return false;
        }

        print_list();

        return true;
    }

    /********************************************************/
    void print_list()
    {
        yInfo() << "Loaded motion list:\n";
        for (map<string, listEntry>::iterator it = motion_list.begin(); it!= motion_list.end(); it++)
        {
            listEntry &entry = it->second;
            yInfo() << "[" << it->first << "]";
            yInfo() << "Tag = " << entry.tag;
            yInfo() << "id joint = " << entry.id_joint;
            yInfo() << "Min = " << entry.min;
            yInfo() << "Max = " << entry.max << "\n";
        }
    }

    /********************************************************/
    bool getJoint(string &joint, listEntry &entry)
    {
        joint = entry.tag;
        return true;
    }

    /********************************************************/
    bool getSkeleton(string joint_name)
    {
        if(opcPort.getOutputCount() > 0)
        {
            yInfo() << "Get skeleton value for" << joint_name;

            //ask for the property id
            Bottle cmd, reply;
            cmd.addVocab(Vocab::encode("ask"));
            Bottle &content = cmd.addList().addList();
            content.addString("body");

//            yInfo() << "Query opc: " << cmd.toString();
            opcPort.write(cmd, reply);
//            yInfo() << "Reply from opc:" << reply.toString();

            if(reply.size() > 1)
            {
                if(reply.get(0).asVocab() == Vocab::encode("ack"))
                {
                    if(Bottle *idField = reply.get(1).asList())
                    {
                        if(Bottle *idValues = idField->get(1).asList())
                        {
                            int id = idValues->get(6).asInt();
//                            yInfo() << id;

                            //given the id, get the value of the property
                            cmd.clear();
                            cmd.addVocab(Vocab::encode("get"));
                            Bottle &content = cmd.addList().addList();
                            content.addString("id");
                            content.addInt(id);
                            Bottle replyProp;

//                            yInfo() << "Command sent to the port: " << cmd.toString();
                            opcPort.write(cmd, replyProp);
//                            yInfo() << "Reply from opc:" << replyProp.toString();

                            if(replyProp.get(0).asVocab() == Vocab::encode("ack"))
                            {
                                if(Bottle *propField = replyProp.get(1).asList())
                                {
                                    if(Bottle *propSubField = propField->find("body").asList())
                                    {
//                                       yInfo() << propSubField->get(0).asString();
//                                        yInfo() << joint_name+"Right";
                                       if(Bottle *joint_3d = propSubField->find(joint_name+"Right").asList())
                                       {
                                           j_3d.resize(3);
                                           j_3d[0] = joint_3d->get(0).asDouble();
                                           j_3d[1] = joint_3d->get(1).asDouble();
                                           j_3d[2] = joint_3d->get(2).asDouble();
                                           yInfo() << j_3d[0] << " " << j_3d[1] << " " << j_3d[2];
                                       }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /********************************************************/
    bool attach(RpcServer &source)
    {
        return this->yarp().attachAsServer(source);
    }

public:
    /********************************************************/
    bool configure(ResourceFinder &rf)
    {
        this->rf = &rf;
        string moduleName = rf.check("name", Value("motionAnalyzer")).asString();
        setName(moduleName.c_str());

        string robot = rf.check("robot", Value("icub")).asString();

        opcPort.open(("/" + getName() + "/opc").c_str());
        rpcPort.open(("/" + getName() + "/cmd").c_str());
        attach(rpcPort);

        if(!load())
            return false;

        return true;
    }

    /********************************************************/
    bool close()
    {
        opcPort.close();
        rpcPort.close();

        return true;
    }

    /********************************************************/
    double getPeriod()
    {
        return 1.0;
    }

    /********************************************************/
    bool updateModule()
    {
        string joint_name;
        for (map<string, listEntry>::iterator it = motion_list.begin(); it!= motion_list.end(); it++)
        {
            listEntry &entry = it->second;
            getJoint(joint_name, entry);

            getSkeleton(joint_name);
        }

        return true;
    }

};

int main(int argc, char *argv[])
{
    Network yarp;
    if(!yarp.checkNetwork())
    {
        yError() << "YARP server not available";
        return -1;
    }

    Manager manager;
    ResourceFinder rf;

    rf.setVerbose();
    rf.setDefaultContext("motionAnalyzer");
//    rf.setDefaultConfigFile("motion-list.ini");
    rf.setDefault("configuration-file", "motion-list.ini");

    rf.configure(argc, argv);


    return manager.runModule(rf);

    return 0;
}
