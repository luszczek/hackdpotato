#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <fstream>

void exitWithError(const std::string &error) 
{
	std::cout << error;
	std::cin.ignore();
	std::cin.get();

	exit(EXIT_FAILURE);
}
