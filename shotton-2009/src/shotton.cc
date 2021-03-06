#include <shotton/shotton.h>

namespace lab1231_sun_prj {
namespace shotton {

void train(DataParam data_param, EnergyParam* energy_param) {
  std::cout << "train(): BEGIN\n";

  const std::string data_name = data_param["name"];

  if (data_param["name"]=="MSRC") {
    // From [Shotton, 2009                                                 ]
    (*energy_param)["theta_phi_1"] = 4.5;
    (*energy_param)["theta_phi_2"] = 1.0;  
  } 
  else {
    assert(false && "Unknown dataset!");
  }

  std::cout << "train(): END\n";
}

Eigen::MatrixXi annotate(const std::string& img_filename, DataParam data_param, EnergyParam energy_param) {
  using namespace std;
  std::cout << "annotate(): BEGIN\n";

  const string img_path = string(data_param["ori_img_dir"] + img_filename);
  cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

  const size_t n_var = img.rows * img.cols;
  const size_t n_label = boost::lexical_cast<size_t>(data_param["n_label"]);

  GraphicalModel gm( opengm::SimpleDiscreteSpace<size_t, size_t>(n_var, n_label) );

  set_1st_order(img, img_filename, n_label, &gm);
  set_2nd_order(img, n_label, energy_param, &gm);

  Eigen::MatrixXi ann(img.rows, img.cols);
  const string method = "AlphaExpansionFusion";//: "AlphaExpansion", "ICM"
  infer(method, gm, n_var, &ann);

  std::cout << "annotate(): END\n";
  return ann;
}

void infer(const std::string& method, const GraphicalModel& gm, const size_t& n_var, Eigen::MatrixXi* ann) {
  using namespace std;
  cout << "infer(): BEGIN\n";
  cout << "method= " << method << endl;

  vector<size_t> ann_vec(n_var);
  
  if (method=="AlphaExpansion") {
    typedef 
    opengm::external::MinSTCutKolmogorov<size_t, double> 
    MinStCutType;

    typedef 
    opengm::GraphCut<GraphicalModel, opengm::Minimizer, MinStCutType> 
    MinGraphCut;
    
    typedef 
    opengm::AlphaExpansion<GraphicalModel, MinGraphCut> 
    MinAlphaExpansion;

    MinAlphaExpansion inf_engine(gm);

    cout << "Inferring ..." << endl;
    inf_engine.infer();
    inf_engine.arg(ann_vec);
  }
  else if (method=="AlphaExpansionFusion") {
    typedef 
    opengm::external::MinSTCutKolmogorov<size_t, double> 
    MinStCutType;

    typedef 
    opengm::GraphCut<GraphicalModel, opengm::Minimizer, MinStCutType> 
    MinGraphCut;
    
    typedef 
    opengm::AlphaExpansionFusion<GraphicalModel, MinGraphCut> 
    MinAlphaExpansionFusion;

    MinAlphaExpansionFusion inf_engine(gm);
  }
  else if (method=="ICM") {
    typedef opengm::ICM<GraphicalModel, opengm::Minimizer> IcmType;
    IcmType::VerboseVisitorType visitor;

    IcmType inf_engine(gm);
    inf_engine.infer(visitor);
  }
  else {
    assert(false && "Unknown inference method");
  }

  //
  size_t idx = 0;
  for (size_t i=0; i<ann->rows(); ++i) {
    for (size_t j=0; j<ann->cols(); ++j) {
      (*ann)(i,j) = ann_vec.at(idx);
      ++ idx;
    }
  }

  std::cout << "infer(): END\n";
}

void set_1st_order(const cv::Mat& img, const std::string& img_filename, const size_t& n_label, GraphicalModel* gm) {
  using namespace std;

  const string unary_prob_img_dir = "/media/tor/423AF0113AF003A7/tor/robotics/prj/011/dataset/msrc/unary_philipp/msrc_compressed/";
  const string unary_prob_img_path = string( unary_prob_img_dir+img_filename.substr(0,img_filename.size()-4)+".c_unary" );// -4 for .bmp
  cout << "unary_prob_img_path= " << unary_prob_img_path << endl;

  ProbImage unary_prob_img;
  unary_prob_img.decompress( unary_prob_img_path.c_str() );

  assert(unary_prob_img.width()==img.cols && "err");
  assert(unary_prob_img.height()==img.rows && "err");

  for (size_t x=0; x<img.cols; ++x) {
    for (size_t y=0; y<img.rows; ++y) {
      // add a function
      const size_t shape[] = {n_label};
      opengm::ExplicitFunction<float> energy(shape, shape+1);

      for(int i = 0; i < n_label; i++) 
        energy(i) = -1 * unary_prob_img(x,y,i);

      GraphicalModel::FunctionIdentifier fid = gm->addFunction(energy);
      
      // add a factor
      size_t var_idxes[] = {util::var_idx(x, y, img.cols)};
      gm->addFactor(fid, var_idxes, var_idxes+1);
    }
  }
}

void set_2nd_order(const cv::Mat& img, const size_t& n_label, EnergyParam energy_param, GraphicalModel* gm) {
  // Params needed by the Pott model
  const float equal_pen = 0.0;

  //
  float beta;
  beta = edge_potential::get_beta(img);

  Eigen::MatrixXd theta_phi(2, 1);
  theta_phi << energy_param["theta_phi_1"], 
               energy_param["theta_phi_2"];

  //
  for (size_t x=0; x<img.cols; ++x) {
    for (size_t y=0; y<img.rows; ++y) {
      cv::Point2i p1;   
      p1.x = x; p1.y = y;

      // (x, y) -- (x + 1, y)
      if (x+1 < img.cols) {
        // add a function
        cv::Point2i p2;   
        p2.x = x+1; p2.y = y;

        float unequal_pen;
        unequal_pen = edge_potential::potential(img.at<cv::Vec3b>(p1), img.at<cv::Vec3b>(p2), beta, theta_phi);

        //
        opengm::PottsFunction<float> pott(n_label, n_label, equal_pen, unequal_pen);
        GraphicalModel::FunctionIdentifier fid = gm->addFunction(pott);

        // add a factor
        size_t var_idxes[] = {util::var_idx(x,y,img.cols), util::var_idx(x+1,y,img.cols)};
        std::sort(var_idxes, var_idxes + 2);
        gm->addFactor(fid, var_idxes, var_idxes + 2);
      }

      // (x, y) -- (x, y + 1)
      if (y+1 < img.rows) {
        // add a function
        cv::Point2i p2;   
        p2.x = x; p2.y = y+1;

        float unequal_pen;
        unequal_pen = edge_potential::potential(img.at<cv::Vec3b>(p1), img.at<cv::Vec3b>(p2), beta, theta_phi);

        //
        opengm::PottsFunction<float> pott(n_label, n_label, equal_pen, unequal_pen);
        GraphicalModel::FunctionIdentifier fid = gm->addFunction(pott);

        // add a factor
        size_t var_idxes[] = {util::var_idx(x,y,img.cols), util::var_idx(x,y+1,img.cols)};
        std::sort(var_idxes, var_idxes + 2);
        gm->addFactor(fid, var_idxes, var_idxes + 2);
      }
    }
  }
}

}// namespace shotton
} // namespace lab1231_sun_prj