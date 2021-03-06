// MotionLearn.cpp : Defines the entry point for the console application.
#include "MIO.h"
#include "lib/Eigen/Core"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/**
 * prototypes
 **/
static void ShowUsage(void);

static void CheckOption(char *option, int argc, int minargc);

bool verbose = false;

int main(int argc, char *argv[]) {
  // first argument is program name
  argv++, argc--;

  // look for help
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "-help")) {
      ShowUsage();
    }
  }

  // apply options
  for (int i = 0; i < argc; i++) {
    if (!strcmp(*argv, "-v")) {
      verbose = true;
      argv += 1, argc -= 1;
    }
  }

  // no argument case
  if (argc == 0) {
    ShowUsage();
  }

  // parse arguments
  while (argc > 0) {
    if (**argv == '-') {
      if (!strcmp(*argv, "-loadData")) {
        int numopts = 1;

        // numopts+1 because parameter name itself counts
        CheckOption(*argv, argc, numopts + 1);

        // TODO: call network function
        std::cout << "Attempting to read file " << argv[1] << std::endl;
        Eigen::MatrixXd rawData = matrixFromFile(argv[1], 0, ',');
        std::cout << "Read in " << rawData.rows() << "x" << rawData.cols()
                  << "raw file " << std::endl;
        Eigen::MatrixXd data = rawData.topRightCorner(100, 784);
        std::cout << "Read in " << data.rows() << " x " << data.cols()
                  << "matrix." << std::endl;
        argv += numopts + 1, argc -= numopts + 1;

        int nHidden = 100;
        int nClasses = 10;
        Eigen::MatrixXd inputToHidden = Eigen::MatrixXd::Random(784, nHidden);
        Eigen::MatrixXd hiddenToOutput =
            Eigen::MatrixXd::Random(nHidden, nClasses);

        // for first 10 rows
        for (int i = 0; i < 5; i++) {
          Eigen::VectorXd hiddenLayer = (data.row(i) * inputToHidden);

          // apply RELU
          for (int j = 0; j < hiddenLayer.cols(); j++) {
            hiddenLayer[j] = hiddenLayer[j] < 0 ? 0 : hiddenLayer[j];
          }

          Eigen::VectorXd output = (hiddenLayer.transpose() * hiddenToOutput);
          output =
              output - Eigen::VectorXd::Ones(output.size()) * output.maxCoeff();
          softmax(output);
          // prediction is whatever class has max value
          std::cout << "row " << i << " prediction:" << std::endl;
          std::cout << output.transpose() << std::endl;
          std::cout << "ground truth: " << rawData(i, 0) << std::endl;
        }
      } else if (!strcmp(*argv, "-forwardProp")) {
        int numopts = 0;

        // numopts+1 because parameter name itself counts
        CheckOption(*argv, argc, numopts + 1);

        // TODO: call network function
        argv += numopts + 1, argc -= numopts + 1;
      } else {
        fprintf(stderr, "invalid option: %s\n", *argv);
        ShowUsage();
      }
    } else {
      fprintf(stderr, "DeepNav: invalid option (2): %s\n", *argv);
      ShowUsage();
    }
  }

  return EXIT_SUCCESS;
}

/**
 * ShowUsage
 **/

static char options[] = "-help (show this message)\n"
                        "-v verbose output\n"
                        "- forwardProp\n";

static void ShowUsage(void) {
  fprintf(stderr, "Usage: DeepNav [-option [arg ...] ...] -output \n");
  fprintf(stderr, "%s", options);
  exit(EXIT_FAILURE);
}

/**
 * CheckOption
 **/
static void CheckOption(char *option, int argc, int minargc) {
  if (argc < minargc) {
    fprintf(stderr, "Too few arguments for %s, expected %d, received %d\n",
            option, minargc, argc);
    ShowUsage();
  }
}
