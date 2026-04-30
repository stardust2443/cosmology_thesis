#include "CBL.h"
#include <iostream>
#include <string>
#include <vector>

int main (int argc, char **argv) {
  try {
    // 1. Enforce Command Line Arguments
    if (argc != 5) {
        std::cerr << "Usage: ./run_aletheia_recon <input_prefix> <output_prefix> <start_idx> <end_idx>" << std::endl;
        return 1;
    }

    std::string input_prefix  = argv[1];
    std::string output_prefix = argv[2];
    int start_idx             = std::stoi(argv[3]);
    int end_idx               = std::stoi(argv[4]);

    std::cout << "--- Starting BitVF Recon from realization " << start_idx << " to " << end_idx << " ---" << std::endl;

    // 2. Hardcoded Physics Parameters (Thesis Sec 6.5.2)
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

    // Loop over the realizations assigned to this array task
    for (int i = start_idx; i <= end_idx; ++i) {
        std::string tracer_file = input_prefix + "_R" + std::to_string(i) + ".dat";
        std::string output_file = output_prefix + "_R" + std::to_string(i);

        // We can vary the seed by realization index so each is distinct if needed, 
        // though Nrec=50 inside BitVF handles its own internal stochasticity.
        unsigned int seed = 42 + i; 

        std::cout << "\n[Realization " << i << "] Running Optimal Transport and identifying underdensities..." << std::endl;
        std::cout << "  -> Input : " << tracer_file << std::endl;
        std::cout << "  -> Output: " << output_dir << output_file << std::endl;

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
                 
        std::cout << "[Realization " << i << "] BitVF Run Complete." << std::endl;
    }
    
    std::cout << "\n--- Array Job Complete ---" << std::endl;
  }
  
  catch (cbl::glob::Exception &exc) { 
      std::cerr << exc.what() << std::endl; 
      exit(1); 
  }
   
  return 0; 
}
