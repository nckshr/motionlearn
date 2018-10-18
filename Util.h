#pragma once
#include "lib/Eigen/Core"
#include <algorithm>
#include <cmath>
#include <list>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
template <class T> T getAt(std::list<T> source, int ix) {
  T toReturn;
  int i = 0;
  typename std::list<T>::iterator iter = source.begin();
  while (iter != source.end() && i < ix) {
    i++;
    ++iter;
  }
  toReturn = (*iter);
  return toReturn;
};

typedef std::pair<int, double> intDoublePair;
bool comparator(const intDoublePair &l, const intDoublePair &r);
std::vector<intDoublePair> sortWithIndexReturn(std::vector<double> toSort);
std::vector<intDoublePair> sortWithIndexReturn(std::list<double> toSort);
void softmax(Eigen::VectorXd &x);
void rotate2D(double &vx, double &vy, double theta);
void normalize(double &vx, double &vy);
int argmin(std::vector<double> &x);
std::vector<std::string> split_string(std::string s, char delim);
// Eigen::VectorXd fromStdVector(std::vector<double> x) {
//  return Eigen::VectorXd(x.data());
//};
