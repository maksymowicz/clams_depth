#include <clams_ros/clams_depth.h>
#include <sensor_msgs/image_encodings.h>

#include <string>
#include <stdlib.h>

using namespace std;

typedef image_transport::SubscriberStatusCallback itSSCb;
typedef ros::SubscriberStatusCallback rosSSCb;

ClamsDepth::ClamsDepth(ros::NodeHandle nh_, ros::NodeHandle pnh_) : 
    nh(nh_), pnh(pnh_), it(nh_)
{
    lock_guard<mutex> lock(connect_mutex);

    itSSCb itssc = bind(&ClamsDepth::connectCb, this);
    rosSSCb rssc = bind(&ClamsDepth::connectCb, this);
    depth_pub = it.advertiseCamera("depth_out", 1, itssc, itssc, rssc, rssc);

    string model_file;
    if (!pnh.getParam("model", model_file))
    {
        ROS_FATAL("ClamsDepth: Please provide the full path and file name "
                "of the discrete distortion model created using CLAMS.");
        exit(EXIT_FAILURE);
    }
    if (clams::fileExtension(model_file).compare("txt") != 0)
        ROS_WARN("ClamsDepth: discrete distortion models saved in formats "
                "other than ASCII (*.txt) are not very portable. Attempting "
                "to open anyway...");

    model.load(model_file);
    if (!model.isValid())
    {
        ROS_FATAL("ClamsDepth: Loaded discrete distortion model invalid.");
        exit(EXIT_FAILURE);
    }
    ROS_INFO("ClamsDepth: using distortion model from %s.", model_file.c_str());
}

ClamsDepth::~ClamsDepth()
{
    depth_sub.shutdown();
}

void ClamsDepth::connectCb()
{
    lock_guard<mutex> lock(connect_mutex);
    if (!depth_sub && depth_pub.getNumSubscribers() > 0)
    {
        ROS_INFO("ClamsDepth: Connecting to depth topic.");
        image_transport::TransportHints h("raw", ros::TransportHints(), pnh);
        depth_sub = it.subscribeCamera("depth_in", 1, &ClamsDepth::depthCb, 
                this, h);
    }
}

void ClamsDepth::disconnectCb()
{
    lock_guard<mutex> lock(connect_mutex);
    if (depth_pub.getNumSubscribers() == 0)
    {
        ROS_INFO("ClamsDepth: Unsubscribing from depth topic.");
        depth_sub.shutdown();
    }
}

void ClamsDepth::depthCb(const sensor_msgs::ImageConstPtr& img, 
        const sensor_msgs::CameraInfoConstPtr& ci)
{
    static sensor_msgs::CameraInfoPtr info(new sensor_msgs::CameraInfo(*ci));
    info->header.stamp = ci->header.stamp;

    sensor_msgs::Image::Ptr depth(new sensor_msgs::Image(*img));

    model.undistort(depth);
    depth_pub.publish(depth, info);
}
