#ifndef clams_depth_h
#define clams_depth_h

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <clams/discrete_depth_distortion_model.h>
#include <mutex>

class ClamsDepth
{
    public:
        ClamsDepth(ros::NodeHandle&, ros::NodeHandle&);
        ~ClamsDepth();

    private:
        ros::NodeHandle nh, pnh;
        image_transport::ImageTransport it;
        image_transport::CameraSubscriber depth_sub;
        image_transport::CameraPublisher depth_pub;

        clams::DiscreteDepthDistortionModel model;

        std::mutex connect_mutex;

        void connectCb();
        void disconnectCb();
        void depthCb(const sensor_msgs::ImageConstPtr& img, 
                const sensor_msgs::CameraInfoConstPtr& info);
};

#endif
