# Cross validation Sampling Julia Codes

This repo has implemented 1d, 2d, nd cross validation sampling techniques using Julia and tests with several benchmark functions


Current Implementations:
- 1d sampling: Gramacylee Function
- 2d sampling: 2d Rosenbrock Function
- nd sampling: 8d Waterflow Function, 3d Weldedbeam Function


File Structures:
- Root directory: Latest codes for 1d, 2d, nd sampling
- Results folder: All the results and graphs generated from the codes shall go in this folder
- Tests folder: All variations of codes with different hyperparameters shall go in this folder. A place to hold files that need to test or has been tested on hpc.


File Names:

File names for nd weldedbeam sampling results are structured as -- "ndsampling_weldedbeam_MSE_<total # of sampling points>p_<initial # of sampling points>i_<# of target sampling points in each iteration>n"


NOTE: There are places with NOTE comments in the codes that need future works.