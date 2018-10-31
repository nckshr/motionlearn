#include <vector>
#include <string>
#include <Eigen/Core>

enum LayerType {
  Pass = 0,
  RELU = 1,
  SoftMax = 2
};

enum CostType {
  CrossEntropy = 0
};

class BasicNN {
public:
  BasicNN(std::vector<std::pair<LayerType,int>> _layers, CostType _cost_type);
  BasicNN(std::string file_name);

  void SaveToFile(std::string file_name);
  void LoadFromFile(std::string file_name);

  int Classify(const Eigen::VectorXf& example) const;
  int NCorrectClassified(const Eigen::MatrixXf& data, const Eigen::VectorXi& labels) const;

  float ComputeCost(const Eigen::MatrixXf& data, const Eigen::VectorXi& labels, int* n_correct = nullptr) const;
  float ComputeCostBatch(const Eigen::MatrixXf& data, const Eigen::VectorXi& labels, int mini_batch_size, int* n_correct = nullptr) const;
  void UpdateWeights(const Eigen::MatrixXf& data, const Eigen::VectorXi& labels, float alpha);
  void UpdateWeightsBatch(const Eigen::MatrixXf& data, const Eigen::VectorXi& labels, float alpha, int mini_batch_size);

private:
  std::vector<std::pair<LayerType,int>> layers;
  std::vector<Eigen::MatrixXf> weights;
  CostType cost_type;

  Eigen::VectorXf ComputeSoftMax(const Eigen::VectorXf& v) const;
  void ApplyLayer(Eigen::MatrixXf& vals, const LayerType type) const;
  void ApplyLayerGradient(Eigen::MatrixXf& dc_dli, LayerType type, const Eigen::MatrixXf& int_values, const Eigen::VectorXi& labels) const;
  Eigen::MatrixXf ForwardProp(const Eigen::MatrixXf& data, std::vector<Eigen::MatrixXf>* intermediate_values = nullptr) const;
};
