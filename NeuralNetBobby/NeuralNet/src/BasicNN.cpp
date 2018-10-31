#include <NN/BasicNN.h>
#include <utils/utils.h>
#include <iostream>
#include <fstream>

BasicNN::BasicNN(std::vector<std::pair<LayerType,int>> _layers, CostType _cost_type) {
  layers = _layers;
  cost_type = _cost_type;

  for (size_t i = 0; i < layers.size()-1; ++i) {
    int dim_1 = layers[i].second;
    int dim_2 = layers[i+1].second;

    Eigen::MatrixXf W = Eigen::MatrixXf::Random(dim_2,dim_1);
    weights.push_back(W);
  }
}

BasicNN::BasicNN(std::string file_name) {
  LoadFromFile(file_name);
}

void BasicNN::LoadFromFile(std::string file_name) {
  layers.clear();
  weights.clear();
  std::vector<std::vector<std::string>> data_vec = LoadFileByToken(file_name, 0, ',');
  // Read in layer information
  assert(data_vec.size() >= 2);
  assert(data_vec[0].size() % 2 == 0);
  for (size_t i = 0; i < data_vec[0].size(); i+=2) {
    int layer_type = std::atoi(data_vec[0][i].c_str());
    int layer_size = std::atoi(data_vec[0][i+1].c_str());
    layers.emplace_back(static_cast<LayerType>(layer_type),layer_size);
  }
  assert(data_vec[1].size() == 1);
  int cost_type_int = std::atoi(data_vec[1][0].c_str());
  cost_type = static_cast<CostType>(cost_type_int);

  // Read in weights
  for (size_t i = 0; i < layers.size()-1; ++i) {
    int dim_1 = layers[i].second;
    int dim_2 = layers[i+1].second;

    Eigen::MatrixXf W = Eigen::MatrixXf::Zero(dim_2,dim_1);
    weights.push_back(W);
  }
  int file_row = 2;
  for (size_t i = 0; i < weights.size(); ++i) {
    for (int j = 0; j < weights[i].cols(); ++j) {
      assert(static_cast<int>(data_vec[file_row].size()) == weights[i].rows());
      for (int k = 0; k < weights[i].rows(); ++k) {
        weights[i](k,j) = std::atof(data_vec[file_row][k].c_str());
      }
      file_row++;
    }
  }
  
  std::ifstream net_file(file_name);  
}

void BasicNN::SaveToFile(std::string file_name) {
  std::ofstream net_file(file_name);

  // Write out layer information
  bool first = true;
  for (std::pair<LayerType,int> layer : layers) {
    if (!first) {
      net_file << ",";
    } else {
      first = false;
    }
    net_file << static_cast<int>(layer.first) << "," << layer.second;
  }
  net_file << std::endl;
  net_file << static_cast<int>(cost_type) << std::endl;
  
  // write out weight matricies
  for (size_t i = 0; i < weights.size(); ++i) {
    for (int j = 0; j < weights[i].cols(); ++j) {
      for (int k = 0; k < weights[i].rows(); ++k) {
        net_file << weights[i](k,j);
        if (k != weights[i].rows()-1) {
          net_file << ",";
        }
      }
      net_file << std::endl;
    }
  }

  net_file.close();
}

Eigen::VectorXf BasicNN::ComputeSoftMax(const Eigen::VectorXf& v) const {
  Eigen::VectorXf out = v;
  float max_v = out.maxCoeff();
  out.array() -= max_v;
  out = out.array().exp();
  out /= out.sum();
  return out;
}

void BasicNN::ApplyLayer(Eigen::MatrixXf& vals, const LayerType type) const {
  if (type == LayerType::RELU) {
    vals = vals.array().max(0.0f);
  } else if (type == LayerType::SoftMax) {
    for (int i = 0; i < vals.cols(); ++i) {
      vals.col(i) = ComputeSoftMax(vals.col(i));
    }
  } else if (type == LayerType::Pass) {
    // Do nothing
  } else {
    std::cerr << "Layer Type Not Implemented!" << std::endl;
    exit(-1);
  }
}

Eigen::MatrixXf BasicNN::ForwardProp(const Eigen::MatrixXf& data, std::vector<Eigen::MatrixXf>* intermediate_values) const {
  Eigen::MatrixXf curr_vals = data;
  // Save results, if needed
  if (intermediate_values != nullptr) {
    intermediate_values->push_back(curr_vals);
  }
  for (size_t i = 0; i < layers.size()-1; ++i) {
    // Apply Weights:
    curr_vals.applyOnTheLeft(weights[i]); // = weights[i] * curr_vals;
    // Apply Non-linearity
    ApplyLayer(curr_vals, layers[i+1].first);

    // Save results, if needed
    if (intermediate_values != nullptr) {
      intermediate_values->push_back(curr_vals);
    }
  }
  return curr_vals;
}

int BasicNN::Classify(const Eigen::VectorXf& example) const {
  Eigen::MatrixXf probs = ForwardProp(example);
  int r,c;
  probs.maxCoeff(&r,&c);
  return r;
}

int BasicNN::NCorrectClassified(const Eigen::MatrixXf& data, const Eigen::VectorXi& labels) const {
  int n_correct = 0;
  for (int i = 0; i < labels.rows(); ++i) {
    int guess = Classify(data.col(i));
    if (guess == labels[i]) {++n_correct;}
  }
  return n_correct;
}

float BasicNN::ComputeCost(const Eigen::MatrixXf& data, const Eigen::VectorXi& labels, int* n_correct) const {
  Eigen::MatrixXf results = ForwardProp(data);
  float cost = 0.0f;
  if (n_correct != nullptr) { (*n_correct) = 0; }
  for (int i = 0; i < results.cols(); ++i) {
    float p_i = results(labels[i], i);
    cost += std::log(std::max(p_i, 1e-15f));
    if (n_correct != nullptr) {
      int r,c;
      results.col(i).maxCoeff(&r,&c);
      if (labels[i] == r) { (*n_correct) += 1; }
    }
  }
  return -cost / results.cols();
}

float BasicNN::ComputeCostBatch(const Eigen::MatrixXf& data, const Eigen::VectorXi& labels, int mini_batch_size, int* n_correct) const {
  int n_data = static_cast<int>(labels.size());
  int n_batches = static_cast<int>(std::ceil(static_cast<float>(n_data) / static_cast<float>(mini_batch_size)));
  float total_cost = 0.0f;
  if (n_correct != nullptr) { (*n_correct) = 0; }
  for (int j = 0; j < n_batches; ++j) {
    int n_to_grab = ((j+1)*mini_batch_size < n_data) ? mini_batch_size : n_data - j*mini_batch_size;
    const Eigen::MatrixXf& data_seg = data.middleCols(j*mini_batch_size, n_to_grab);
    const Eigen::VectorXi& labels_seg = labels.segment(j*mini_batch_size, n_to_grab);
    float batch_cost;
    if (n_correct != nullptr) {
      int n_correct_batch;
      batch_cost = ComputeCost(data_seg, labels_seg, &n_correct_batch);
      (*n_correct) += n_correct_batch;
    } else {
      batch_cost = ComputeCost(data_seg, labels_seg);
    }
    float batch_weight = static_cast<float>(n_to_grab) / static_cast<float>(n_data);
    total_cost += batch_cost * batch_weight;
  }
  return total_cost;
}

void BasicNN::ApplyLayerGradient(Eigen::MatrixXf& dc_dli, LayerType type, const Eigen::MatrixXf& int_values, const Eigen::VectorXi& labels) const {
  if (type == LayerType::SoftMax) {
    for (int i = 0; i < labels.size(); ++i) {
      dc_dli(labels[i],i) -= 1.0f;
    }
  } else if (type == LayerType::RELU) {
    dc_dli.noalias() = (int_values.array()>0.0f).select(dc_dli,0.0f);
  } else if (type == LayerType::Pass) {
    // Do nothing -> layer is identity
  } else {
    std::cerr << "Layer Type Gradient not implemented!" << std::endl;
    exit(-1);
  }
}

void BasicNN::UpdateWeights(const Eigen::MatrixXf& data, const Eigen::VectorXi& labels, float alpha) {
  std::vector<Eigen::MatrixXf> intermediate_values;
  ForwardProp(data, &intermediate_values);

  std::vector<Eigen::MatrixXf> weights_gradients;
  Eigen::MatrixXf dc_dli = intermediate_values.back();
  for (int i = static_cast<int>(weights.size())-1; i >= 0; --i) {
    // Apply Layer Gradient:
    ApplyLayerGradient(dc_dli, layers[i+1].first, intermediate_values[i+1], labels);

    // Compute Weight Gradient
    weights_gradients.push_back(dc_dli * intermediate_values[i].transpose());
    
    // Apply gradient across weights
    dc_dli = weights[i].transpose() * dc_dli;
  }
  std::reverse(weights_gradients.begin(), weights_gradients.end());

  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = weights[i] - (alpha / labels.rows()) * weights_gradients[i];
  }
}

void BasicNN::UpdateWeightsBatch(const Eigen::MatrixXf& data, const Eigen::VectorXi& labels, float alpha, int mini_batch_size) {
  int n_data = static_cast<int>(labels.size());
  int n_batches = static_cast<int>(std::ceil(static_cast<float>(n_data) / static_cast<float>(mini_batch_size)));
  for (int j = 0; j < n_batches; ++j) {
    int n_to_grab = ((j+1)*mini_batch_size < n_data) ? mini_batch_size : n_data - j*mini_batch_size;
    const Eigen::MatrixXf& data_seg = data.middleCols(j*mini_batch_size, n_to_grab);
    const Eigen::VectorXi& labels_seg = labels.segment(j*mini_batch_size, n_to_grab);
    UpdateWeights(data_seg, labels_seg, alpha);
  }
}
