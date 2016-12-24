#include <clams/discrete_depth_distortion_model.h>
#include <eigen_extensions/eigen_extensions.h>

using namespace std;
using namespace Eigen;

namespace clams
{

  DiscreteFrustum::DiscreteFrustum(int smoothing, double bin_depth) :
    max_dist_(10),
    bin_depth_(bin_depth)
  {
    num_bins_ = ceil(max_dist_ / bin_depth_);
    counts_ = VectorXf::Ones(num_bins_) * smoothing;
    total_numerators_ = VectorXf::Ones(num_bins_) * smoothing;
    total_denominators_ = VectorXf::Ones(num_bins_) * smoothing;
    multipliers_ = VectorXf::Ones(num_bins_);
  }

  inline int DiscreteFrustum::index(double z) const
  {
    return min(num_bins_ - 1, (int)floor(z / bin_depth_));
  }
  
  inline void DiscreteFrustum::undistort(double* z) const
  {
    *z *= multipliers_.coeffRef(index(*z));
  }

  void DiscreteFrustum::interpolatedUndistort(double* z) const
  {
    int idx = index(*z);
    double start = bin_depth_ * idx;
    int idx1;
    if(*z - start < bin_depth_ / 2)
      idx1 = idx;
    else
      idx1 = idx + 1;
    int idx0 = idx1 - 1;
    if(idx0 < 0 || idx1 >= num_bins_ || counts_(idx0) < 50 || counts_(idx1) < 50) {
      undistort(z);
      return;
    }

    double z0 = (idx0 + 1) * bin_depth_ - bin_depth_ * 0.5;
    double coeff1 = (*z - z0) / bin_depth_;
    double coeff0 = 1.0 - coeff1;
    double mult = coeff0 * multipliers_.coeffRef(idx0) + coeff1 * multipliers_.coeffRef(idx1);
    *z *= mult;
  }
  
  void DiscreteFrustum::deserialize(std::istream& in, bool ascii)
    {
    	if(ascii)
    	{
			eigen_extensions::deserializeScalarASCII(in, &max_dist_);
			eigen_extensions::deserializeScalarASCII(in, &num_bins_);
			eigen_extensions::deserializeScalarASCII(in, &bin_depth_);
			eigen_extensions::deserializeASCII(in, &counts_);
			eigen_extensions::deserializeASCII(in, &total_numerators_);
			eigen_extensions::deserializeASCII(in, &total_denominators_);
			eigen_extensions::deserializeASCII(in, &multipliers_);
    	}
    	else
    	{
			eigen_extensions::deserializeScalar(in, &max_dist_);
			eigen_extensions::deserializeScalar(in, &num_bins_);
			eigen_extensions::deserializeScalar(in, &bin_depth_);
			eigen_extensions::deserialize(in, &counts_);
			eigen_extensions::deserialize(in, &total_numerators_);
			eigen_extensions::deserialize(in, &total_denominators_);
			eigen_extensions::deserialize(in, &multipliers_);
    	}
      }

  DiscreteDepthDistortionModel::DiscreteDepthDistortionModel(const DiscreteDepthDistortionModel& other)
  {
    *this = other;
  }

  DiscreteDepthDistortionModel& DiscreteDepthDistortionModel::operator=(const DiscreteDepthDistortionModel& other)
  {
    width_ = other.width_;
    height_ = other.height_;
    bin_width_ = other.bin_width_;
    bin_height_ = other.bin_height_;
    bin_depth_ = other.bin_depth_;
    num_bins_x_ = other.num_bins_x_;
    num_bins_y_ = other.num_bins_y_;
    training_samples_ = other.training_samples_;
  
    frustums_ = other.frustums_;
    for(size_t i = 0; i < frustums_.size(); ++i)
      for(size_t j = 0; j < frustums_[i].size(); ++j)
        frustums_[i][j] = new DiscreteFrustum(*other.frustums_[i][j]);

    return *this;
  }

  DiscreteDepthDistortionModel::DiscreteDepthDistortionModel(int width, int height,
                                                             int bin_width, int bin_height,
                                                             double bin_depth,
                                                             int smoothing) :
    width_(width),
    height_(height),
    bin_width_(bin_width),
    bin_height_(bin_height),
    bin_depth_(bin_depth)
  {
    assert(width_ % bin_width_ == 0);
    assert(height_ % bin_height_ == 0);

    num_bins_x_ = width_ / bin_width_;
    num_bins_y_ = height_ / bin_height_;
  
    frustums_.resize(num_bins_y_);
    for(size_t i = 0; i < frustums_.size(); ++i) {
      frustums_[i].resize(num_bins_x_, NULL);
      for(size_t j = 0; j < frustums_[i].size(); ++j)
        frustums_[i][j] = new DiscreteFrustum(smoothing, bin_depth);
    }

    training_samples_ = 0;
  }

  void DiscreteDepthDistortionModel::deleteFrustums()
  {
    for(size_t y = 0; y < frustums_.size(); ++y)
      for(size_t x = 0; x < frustums_[y].size(); ++x)
        if(frustums_[y][x])
          delete frustums_[y][x];
    training_samples_ = 0;
  }

  DiscreteDepthDistortionModel::~DiscreteDepthDistortionModel()
  {
    deleteFrustums();
  }

  void DiscreteDepthDistortionModel::undistort(cv::Mat & depth) const
  {
    assert(width_ == depth.cols);
    assert(height_ ==depth.rows);
    assert(depth.type() == CV_16UC1 || depth.type() == CV_32FC1);
    if(depth.type() == CV_32FC1)
    {
		#pragma omp parallel for
		for(int v = 0; v < height_; ++v) {
		  for(int u = 0; u < width_; ++u) {
			 float & z = depth.at<float>(v, u);
			if(std::isnan(z) || z == 0.0f)
			  continue;
			double zf = z;
			frustum(v, u).interpolatedUndistort(&zf);
			z = zf;
		  }
		}
    }
    else
    {
		#pragma omp parallel for
		for(int v = 0; v < height_; ++v) {
		  for(int u = 0; u < width_; ++u) {
		    unsigned short & z = depth.at<unsigned short>(v, u);
			if(std::isnan(z) || z == 0)
			  continue;
			double zf = z * 0.001;
			frustum(v, u).interpolatedUndistort(&zf);
			z = zf*1000;
		  }
		}
    }
  }

  string fileExtension(const string& path)
  {
    return path.substr(path.find_last_of(".")+1);
  }

  void DiscreteDepthDistortionModel::load(const std::string& path)
  {
    string file_ext = fileExtension(path);
    bool ascii = file_ext.compare("txt") == 0;

    ifstream f;
    f.open(path.c_str(), ascii ? ios::out : ios::out | ios::binary);
    if(!f.is_open()) {
      cerr << "Failed to open " << path << endl;
      assert(f.is_open());
    }
    deserialize(f, ascii);
    f.close();
  }

  void DiscreteDepthDistortionModel::deserialize(std::istream& in, bool& ascii)
    {
      string buf;
      getline(in, buf);
      assert(buf == "DiscreteDepthDistortionModel v01");
      if(ascii)
      {
    	  eigen_extensions::deserializeScalarASCII(in, &width_);
		  eigen_extensions::deserializeScalarASCII(in, &height_);
		  eigen_extensions::deserializeScalarASCII(in, &bin_width_);
		  eigen_extensions::deserializeScalarASCII(in, &bin_height_);
		  eigen_extensions::deserializeScalarASCII(in, &bin_depth_);
		  eigen_extensions::deserializeScalarASCII(in, &num_bins_x_);
		  eigen_extensions::deserializeScalarASCII(in, &num_bins_y_);
		  eigen_extensions::deserializeScalarASCII(in, &training_samples_);
      }
      else
      {
		  eigen_extensions::deserializeScalar(in, &width_);
		  eigen_extensions::deserializeScalar(in, &height_);
		  eigen_extensions::deserializeScalar(in, &bin_width_);
		  eigen_extensions::deserializeScalar(in, &bin_height_);
		  eigen_extensions::deserializeScalar(in, &bin_depth_);
		  eigen_extensions::deserializeScalar(in, &num_bins_x_);
		  eigen_extensions::deserializeScalar(in, &num_bins_y_);
		  eigen_extensions::deserializeScalar(in, &training_samples_);
      }
      deleteFrustums();
      frustums_.resize(num_bins_y_);
      for(size_t y = 0; y < frustums_.size(); ++y) {
        frustums_[y].resize(num_bins_x_, NULL);
        for(size_t x = 0; x < frustums_[y].size(); ++x) {
          frustums_[y][x] = new DiscreteFrustum;
          frustums_[y][x]->deserialize(in, ascii);
        }
      }
    }
  
  DiscreteFrustum& DiscreteDepthDistortionModel::frustum(int y, int x)
  {
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
    int xidx = x / bin_width_;
    int yidx = y / bin_height_;
    return (*frustums_[yidx][xidx]);
  }

  const DiscreteFrustum& DiscreteDepthDistortionModel::frustum(int y, int x) const
  {
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
    int xidx = x / bin_width_;
    int yidx = y / bin_height_;
    return (*frustums_[yidx][xidx]);
  }

}  // namespace clams
