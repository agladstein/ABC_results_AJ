#!/usr/bin/env bash

#set -euo pipefail

OBS=$1

cd /vol_c/ABC_test_genome

DIR=obs${OBS}
echo "cd ${DIR}"
cd ${DIR}
echo "PWD: $PWD"
echo "/vol_c/env/simprily_env/bin/python /vol_c/src/ABC_results_AJ/plot_joint.py ABC_test_genome_estimate_10pls_100ret_model0_jointPosterior_2_3_Obs0.txt ABC_test_genome_estimate_10pls_100ret_model0_MarginalPosteriorCharacteristics.txt results_param_observed.txt"
/vol_c/env/simprily_env/bin/python /vol_c/src/ABC_results_AJ/plot_joint.py ABC_test_genome_estimate_10pls_100ret_model0_jointPosterior_2_3_Obs0.txt ABC_test_genome_estimate_10pls_100ret_model0_MarginalPosteriorCharacteristics.txt results_param_observed.txt
cd /vol_c/ABC_test_genome
echo ""

