#include <utils/utils.h>
#include <sstream>

std::vector<std::string> GetParts(std::string s, char delim) {
  std::stringstream ss(s);
  std::vector<std::string> parts;
  for (std::string part; std::getline(ss, part, delim); ) {
    parts.push_back(std::move(part));
  }
  return parts;
}

std::vector<std::vector<std::string>> LoadFileByToken(std::string file_name, int n_skip, char delim) {
  std::ifstream datafile(file_name);
  std::vector<std::vector<std::string>> data_vec;
  int n_lines_read = 0;
  for (std::string line; std::getline(datafile, line); ) {
    n_lines_read++;
    if (n_lines_read <= n_skip) {
      continue;
    }
    std::vector<std::string> parts = GetParts(line, delim);
    data_vec.push_back(parts);
  }
  datafile.close();
  return data_vec;
}