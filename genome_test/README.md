## Analysis

rsynced Ocelote ABC directory to Atmosphere instance.  
`/rsgrps/mfh4/Ariella/SimPrily_genome_test/ABC`

### 1. Plot joint posterior and calculate probability A > B.
Use the run_plot_join.sh script to run plot_joint.py on all ABC results.

```
cd /vol_c/src/ABC_results_AJ
seq 1 100 | parallel -j 4 genome_test/run_plot_join.sh {}
```
run_plot_join.sh takes the obs iterator as the argument.


plot_joint.py takes 3 arguments, 
1. the joint posterior file
2. the marginal posterior characteristics file
3. the results_param_observed.txt file


### 2. Combine all ABC marginal posterior characteristics files and marginal posterior density files.
Use the script combine_results.py.

```
python genome_test/combine_results.py
```