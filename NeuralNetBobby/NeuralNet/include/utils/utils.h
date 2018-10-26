#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

std::vector<std::vector<std::string>> LoadFileByToken(std::string file_name, int n_skip, char delim);