#include <clams_ros/clams_depth.h>
#include <nodelet/nodelet.h>

class ClamsDepthNodelet : public nodelet::Nodelet
{
    private:
        virtual void onInit()
        {
            cdp.reset(new ClamsDepth(getNodeHandle(), getPrivateNodeHandle()));
        };

        std::shared_ptr<ClamsDepth> cdp;
};

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(ClamsDepthNodelet, nodelet::Nodelet) 
