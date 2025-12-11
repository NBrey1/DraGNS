This branch is for calculating DOS of a pure methane system under Titan temperature (T = 94K) and pressure (P = 1.5 bar) conditions. 

-The 94K directory will have the most recent version of files for calculations.
-The DOSchecks directory has sub directories for checking the DOS with varying sampling parameters:
    -BaseFiles directory has the 6 essential files to run a calculation:
        - methane.pdb is the corrdinate file (data from https://cccbdb.nist.gov/expgeom1x.asp)
        - topol.top is the topology file (data from https://cccbdb.nist.gov/expgeom1x.asp)
        - em.mdp is the energy minimization file
        - nvt.mdp is the temperature equilibration file
        - npt1.mdp is the pressure equilibration file. There are 2 npt files for doing an equilibration in multiple phases to better equilibrate the system (npt2.mdp exists but is not used
        - production.mdp is the production run which outputs md.trr (trajectory file) used to calculate DOS
    -dt_01fs is for checks with a dt = 0.1 fs (dt = 0.0001) with varying nstvout
    -dt_05fs is for checks with a dt = 0.5 fs (dt = 0.0005) with varying nstvout
