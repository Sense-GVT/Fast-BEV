#include <iostream>
#include <vector>
#include "view_transformer.hpp"


static void PrintHelp(char *name) {
    std::cout << "demo for view transformer(2d to 3d) cuda accelueration" << std::endl;
    std::cout << "Usage " << name << " [path to input dir] [path to groundtruth dir]"<< std::endl;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        PrintHelp(argv[0]);
        return -1;
    }
    demo(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
    return 0;
}