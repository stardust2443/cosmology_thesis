#include "Cosmology.h"
#include "LCDM.h"
#include <iostream>
#include <fstream>
#include <iomanip>

int main() {
    try {
        // Instantiate the concrete LCDM class
        cbl::cosmology::LCDM cosm;

        // Set perturbation parameters (Planck 2018 values)
        cosm.set_parameter("sigma8", 0.811);
        cosm.set_parameter("n_spec", 0.966);

        std::cout << "--- CosmoBolognaLib Initialized ---" << std::endl;
        std::cout << "Omega_matter : " << cosm.Omega_matter() << std::endl;
        std::cout << "sigma_8 (z=0): " << cosm.sigma8() << std::endl;

        // Open a file to save the theoretical curves
        std::ofstream outfile("growth_history.txt");
        outfile << "# z \t D(z) \t f(z)" << std::endl;

        // Compute D(z) and f(z) from redshift 0 to 3
        std::cout << "Computing growth history..." << std::endl;
        for (double z = 0.0; z <= 3.0; z += 0.1) {
            double D = cosm.DD(z);                     // Linear growth factor
            double f = cosm.linear_growth_rate(z);     // Logarithmic growth rate
            
            outfile << std::fixed << std::setprecision(4);
            outfile << z << "\t" << D << "\t" << f << "\n";
        }
        outfile.close();

        std::cout << "Success! Data written to 'growth_history.txt'" << std::endl;

    } catch (std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}