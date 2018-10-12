#pragma once
#include "Util.h"
#include "lib/Eigen/Core"
#include <fstream>
#include <iostream>
#include <list>
#include <ostream>
#include <sstream>
#include <stdio.h>
#include <string>

const Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
Eigen::MatrixXd matrixFromFile(const char *filename, int skip,
                               char delim = ' ');
Eigen::VectorXd vectorFromFile(const char *filename, int length,
                               bool lenFromHeader);
std::ofstream openFile(const char *file, char mode);
