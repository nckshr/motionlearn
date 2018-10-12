#include "MIO.h"

extern bool verbose;
std::ofstream openFile(const char *file, char mode) {
  std::ofstream f;
  if (mode == 'w') {
    f = std::ofstream(file, std::ofstream::out);
  } else if (mode == 'r') {
    f = std::ofstream(file, std::ofstream::in);
  }
  if (f.fail()) {
    std::cout << "FAILURE: File path not found: " << file << std::endl;
    std::exit(1);
  }
  return f;
}
Eigen::MatrixXd matrixFromFile(const char *filename, int skip, char delim) {
  // std::cout << "Loading Matrix from file" << std::endl;
  Eigen::MatrixXd result;
  std::string line;

  std::fstream f(filename);
  std::list<std::list<double>> rowList;

  int linecount = 0;
  int cols = 0;
  while (std::getline(f, line)) {
    if (skip > 0) {
      skip--;
      continue;
    }
    std::vector<std::string> tokens = split_string(line, delim);
    std::list<double> row;

    for (int i = 0; i < tokens.size(); i++) {
      double val = std::atof(tokens[i].c_str());
      row.push_back(val);
    }
    rowList.push_back(row);
    cols = row.size() > cols ? row.size() : cols;
    linecount++;
    // if (linecount % 1000 == 0) { std::cout << "read " << linecount << "
    // rows..." << std::endl; }
  }
  int rix = 0;
  result = Eigen::MatrixXd(linecount, cols);
  for (std::list<std::list<double>>::iterator liter = rowList.begin();
       liter != rowList.end(); ++liter) {
    int cix = 0;
    for (std::list<double>::iterator diter = (*liter).begin();
         diter != (*liter).end(); ++diter) {
      result(rix, cix) = (*diter);
      cix++;
    }
    rix++;
  }
  return result;
}
Eigen::VectorXd vectorFromFile(const char *filename, int length,
                               bool lenFromHeader) {
  if (fopen(filename, "r") == NULL) {
    std::cout << "WARNING: Target file " << filename
              << " not found!! Result will be empty vector!!" << std::endl;
  }
  Eigen::VectorXd result;
  std::string line;
  std::fstream f(filename);
  if (lenFromHeader) {
    std::getline(f, line);
    std::stringstream ss(line);
    ss >> length;
  }
  result = Eigen::VectorXd(length);
  int linecount = 0;
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    double val;
    ss >> val;
    result[linecount] = val;
    linecount++;
  }
  return result;
}
