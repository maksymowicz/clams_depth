#include <ros/ros.h>
#include <clams_depth/clams_depth.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "clams_depth_node");
    
    ros::NodeHandle nh, pnh("~");
    ClamsDepth cd(nh, pnh);

    ros::spin();

    return 0;
}
