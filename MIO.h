#pragma once
#include "lib/Eigen/Dense"
#include <fstream>
#include <ostream>
#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <list>
#include "Util.h"

const Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
Eigen::MatrixXd matrixFromFile(const char* filename, int skip, char delim=' ');
Eigen::VectorXd vectorFromFile(const char* filename, int length, bool lenFromHeader);
std::ofstream openFile(const char* file, char mode);