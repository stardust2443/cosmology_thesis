#include "CBL.h"
#include <iostream>
#include <string>
#include <vector>

int main (int argc, char **argv) {
  try {
    // 1. Enforce Command Line Arguments
    if (argc != 3) {
        std::cerr << "Usage: ./run_aletheia_bitvf <input_catalog> <output_prefix>" << std::endl;
        return 1;


    std::string tracer_file   = argv[1];
    std::string output_file   = argv[2];

    std::cout << "--- Starting BitVF on " << tracer_file << " ---" << std::endl;

    // 2. Hardcoded Physics Parameters
    std::vector<int> col_trac = {0, 1, 2};
    unsigned int Nrec         = 50;
    double threshold          = 1e-3;
    double sigma              = 0.0;
    double cell_size          = 1.0;
    
    // 3. Output Flags
    bool printDispl           = false;
    bool printDiv             = false;
    bool printGal             = false;
    
    // 4. Formatting & Directories
    std::string output_dir    = "results/";
    char delimiter            = ' ';
    char comment              = '#';
    unsigned int seed         = 42;

    std::cout << "Running Optimal Transport and identifying underdensities..." << std::endl;

    // Execute the finder directly with the variables above
    cbl::bitvf::BitVFbox BitVF(
             tracer_file,
             col_trac,
             Nrec,
             threshold,
             sigma,
             cell_size,
             printDispl,
             printDiv,
             printGal,
             output_dir,
             output_file,
             delimiter,
             comment,
             seed);
    
    std::cout << "\n--- BitVF Run Complete: " << output_file << " ---" << std::endl;
  }
  
  catch (cbl::glob::Exception &exc) { 
      std::cerr << exc.what() << std::endl; 
      exit(1); 
  }
   
  return 0; 
}
}