#include <NN/BasicNN.h>
#include <utils/utils.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <Eigen/Core>
#include <getopt.h>

struct NNArgs {
  bool load_nn = false;
  std::string nn_load_file;
  bool train_data = false;
  std::string train_file;
  bool test_data = false;
  std::string test_file;
  bool save_nn = false;
  std::string nn_save_file;
  int batch_size = 100;
  int n_epochs = 100;
  float alpha = 0.001f;
};

NNArgs ReadParams(int argc, char** argv) {
  int c;
  NNArgs args;

  while (1) {
    static struct option long_options[] = {
      {"trainData", required_argument, 0, 'a'},
      {"testData", required_argument, 0, 'b'},
      {"loadNN", required_argument, 0, 'c'},
      {"saveNN", required_argument, 0, 'd'},
      {"epochs", required_argument, 0, 'e'},
      {"batch", required_argument, 0, 'f'},
      {"alpha", required_argument, 0, 'g'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}
    };
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long (argc, argv, "a:b:c:d:e:f:g:h",
                     long_options, &option_index);
    if (c == -1) { break; }

    switch (c) {
      case 'a':
        args.train_data = true;
        args.train_file = std::string(optarg);
        break;
      case 'b':
        args.test_data = true;
        args.test_file = std::string(optarg);
        break;
      case 'c':
        args.load_nn = true;
        args.nn_load_file = std::string(optarg);
        break;
      case 'd':
        args.save_nn = true;
        args.nn_save_file = std::string(optarg);
        break;
      case 'e':
        args.n_epochs = std::atoi(optarg);
        break;
      case 'f':
        args.batch_size = std::atoi(optarg);
        break;
      case 'g':
        args.alpha = std::atof(optarg);
        break;
      case 'h':
        std::cout << argv[0] << " --trainData train_file --testData test_file --loadNN nn_file_in --saveNN nn_file_out --epochs n_epochs --batch batch_size --alpha alpha" << std::endl;
        exit(0);
        break;
      case '?':
        break;
      default:
        abort();
    }
  }

  return args;
}

void LoadData(std::string file_name, Eigen::MatrixXf& data, Eigen::VectorXi& labels) {
  std::vector<std::vector<std::string>> data_vec = LoadFileByToken(file_name, 1, ',');

  data = Eigen::MatrixXf::Zero(static_cast<int>(data_vec[0].size())-1,static_cast<int>(data_vec.size()));
  labels = Eigen::VectorXi::Zero(static_cast<int>(data_vec.size()));
  for (int i = 0; i < static_cast<int>(data_vec.size()); ++i) {
    labels[i] = std::atoi(data_vec[i][0].c_str());
    for (int j = 1; j < static_cast<int>(data_vec[i].size()); ++j) {
      data(j-1,i) = std::atof(data_vec[i][j].c_str());
    }
  }
}

int main(int argc, char** argv) {
  NNArgs args = ReadParams(argc,argv);

  BasicNN* nn;
  if (args.load_nn) {
    nn = new BasicNN(args.nn_load_file);
  } else {
    std::vector<std::pair<LayerType,int>> layers;
    layers.emplace_back(LayerType::Pass,784);
    layers.emplace_back(LayerType::RELU,50);
    layers.emplace_back(LayerType::SoftMax,10);

    nn = new BasicNN(layers, CostType::CrossEntropy);
  }
  if (args.train_data) {
    Eigen::MatrixXf train_data;
    Eigen::VectorXi train_labels;
    std::cout << "Loading Test Data..." << std::flush;
    LoadData(args.train_file, train_data, train_labels);
    std::cout << "Done!" << std::endl;
    {
      int n_correct;
      float cost = nn->ComputeCostBatch(train_data,train_labels,100, &n_correct);
      std::cout << "Initial:  Train: Number correctly classified: " << n_correct << "  cost: " << cost << std::endl;
    }
    //double cost_init = nn.ComputeCost(data, labels);
    for (int i = 0; i < args.n_epochs; ++i) {
      nn->UpdateWeightsBatch(train_data, train_labels, args.alpha, args.batch_size);
      if (i % 10 == 0) {
        int n_correct;
        float cost = nn->ComputeCostBatch(train_data,train_labels,100, &n_correct);
        std::cout << "Epoch: " << i << " Train: Number correctly classified: " << n_correct << "/" << train_labels.size() << "  cost: " << cost << std::endl;
      }
    }
    {
      int n_correct;
      float cost = nn->ComputeCostBatch(train_data,train_labels,100, &n_correct);
      std::cout << "Final:  Train: Number correctly classified: " << n_correct << "  cost: " << cost << std::endl;
    }
  }
  if (args.test_data) {
    Eigen::MatrixXf test_data;
    Eigen::VectorXi test_labels;
    std::cout << "Loading Test Data..." << std::flush;
    LoadData(args.test_file, test_data, test_labels);
    std::cout << "Done!" << std::endl;
    int n_correct;
    float cost = nn->ComputeCostBatch(test_data,test_labels,100, &n_correct);
    std::cout << "Test: Number correctly classified: " << n_correct << "/" << test_labels.size() << "  cost: " << cost << std::endl;
  }
  if (args.save_nn) {
    nn->SaveToFile(args.nn_save_file);
  }

  return 0;
}
