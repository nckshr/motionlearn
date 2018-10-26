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

  int Classify(const Eigen::VectorXd& example) const;
  int NCorrectClassified(const Eigen::MatrixXd& data, const Eigen::VectorXi& labels) const;

  double ComputeCost(const Eigen::MatrixXd& data, const Eigen::VectorXi& labels, int* n_correct = nullptr) const;
  double ComputeCostBatch(const Eigen::MatrixXd& data, const Eigen::VectorXi& labels, int mini_batch_size, int* n_correct = nullptr) const;
  void UpdateWeights(const Eigen::MatrixXd& data, const Eigen::VectorXi& labels, double alpha);
  void UpdateWeightsBatch(const Eigen::MatrixXd& data, const Eigen::VectorXi& labels, double alpha, int mini_batch_size);

private:
  std::vector<std::pair<LayerType,int>> layers;
  std::vector<Eigen::MatrixXd> weights;
  CostType cost_type;

  Eigen::VectorXd ComputeSoftMax(const Eigen::VectorXd& v) const;
  Eigen::MatrixXd ApplyLayer(const Eigen::MatrixXd& vals, const LayerType type) const;
  void ApplyLayerGradient(Eigen::MatrixXd& dc_dli, LayerType type, const Eigen::MatrixXd& int_values, const Eigen::VectorXi& labels) const;
  Eigen::MatrixXd ForwardProp(const Eigen::MatrixXd& data, std::vector<Eigen::MatrixXd>* intermediate_values = nullptr) const;
};
