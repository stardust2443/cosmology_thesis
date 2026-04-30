#!/bin/bash
cd /Users/arun/projects/my_cosmo_analysis/void_analysis/scripts
echo "--- Running data cleaning ---"
python3 01_clean_catalogs.py

echo ""
echo "--- Running plot generation ---"
python3 04_plot_bitvf_all.py

echo ""
echo "Done!"
