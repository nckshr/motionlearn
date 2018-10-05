// MotionLearn.cpp : Defines the entry point for the console application.


#include "lib/Eigen/Dense"
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

int main(int argc, char* argv[]) {
	// first argument is program name
	argv++, argc--;

	// look for help
	for (int i = 0; i < argc; i++) {
		if (!strcmp(argv[i], "-help")) {
			ShowUsage();
		}
	}
	//apply options 
	for (int i = 0; i < argc; i++) {
		if (!strcmp(*argv, "-v"))
		{
			verbose = true;
			argv += 1, argc -= 1;
		}
		
	}
	// no argument case
	if (argc == 0) {
		ShowUsage();
	}

	// parse arguments
	while (argc > 0)
	{
		if (**argv == '-')
		{

			if (!strcmp(*argv, "-forwardProp"))
			{
				int numopts = 0;
				//numopts+1 because parameter name itself counts
				CheckOption(*argv, argc, numopts+1);
				//TODO: call network function
				argv += numopts+1, argc -= numopts+1;
			}			
			else
			{
				fprintf(stderr, "invalid option: %s\n", *argv);
				ShowUsage();
			}
		}
		else
		{
			fprintf(stderr, "DeepNav: invalid option (2): %s\n", *argv);
			ShowUsage();
		}
	}

	return EXIT_SUCCESS;
}


/**
* ShowUsage
**/
static char options[] =
"-help (show this message)\n"
"-v verbose output\n"
"- forwardProp\n"
;

static void ShowUsage(void)
{
	fprintf(stderr, "Usage: DeepNav [-option [arg ...] ...] -output \n");
	fprintf(stderr, "%s", options);
	exit(EXIT_FAILURE);
}



/**
* CheckOption
**/
static void CheckOption(char *option, int argc, int minargc)
{
	if (argc < minargc)
	{
		fprintf(stderr, "Too few arguments for %s, expected %d, received %d\n", option, minargc, argc);
		ShowUsage();
	}
}
