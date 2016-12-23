#include <clams_depth/clams_depth.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <string>
#include <stdlib.h>

using namespace std;
namespace enc = sensor_msgs::image_encodings;

typedef image_transport::SubscriberStatusCallback itSSCb;
typedef ros::SubscriberStatusCallback rosSSCb;

ClamsDepth::ClamsDepth(ros::NodeHandle& nh_, ros::NodeHandle& pnh_) : 
    nh(nh_), pnh(pnh_), it(nh_)
{
    lock_guard<mutex> lock(connect_mutex);

    itSSCb itssc = bind(&ClamsDepth::connectCb, this);
    rosSSCb rssc = bind(&ClamsDepth::connectCb, this);
    depth_pub = it.advertiseCamera("depth_out", 10, itssc, itssc, rssc, rssc);

    string model_file;
    if (!pnh.getParam("model", model_file))
    {
        ROS_FATAL("ClamsDepth: Please provide the full path and file name of the discrete"
                " distortion model created using CLAMS.");
        exit(EXIT_FAILURE);
    }

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
        ROS_DEBUG("Connecting to depth topic.");
        image_transport::TransportHints h("raw", ros::TransportHints(), pnh);
        depth_sub = it.subscribeCamera("image_in", 10, &ClamsDepth::depthCb, 
                this, h);
    }
}

void ClamsDepth::disconnectCb()
{
    lock_guard<mutex> lock(connect_mutex);
    if (depth_pub.getNumSubscribers() == 0)
    {
        ROS_DEBUG("Unsubscribing from depth topic.");
        depth_sub.shutdown();
    }
}

void ClamsDepth::depthCb(const sensor_msgs::ImageConstPtr& img, 
        const sensor_msgs::CameraInfoConstPtr& ci)
{
    static sensor_msgs::CameraInfoPtr info(new sensor_msgs::CameraInfo(*ci));
    info->header.stamp = ci->header.stamp;

    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(img);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    model.undistort(cv_ptr->image);
    depth_pub.publish(cv_ptr->toImageMsg(), info);
}
