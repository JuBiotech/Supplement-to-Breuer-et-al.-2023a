# Supplement code -to-Breuer-et-al.-2023a

__Spatial discontinuous Galerkin spectral element method for a family of chromatography models in CADET__

Computers \& Chemical Engineering

Jan Michael Breuer<sup>1,2</sup> (ORCID: 0000-0002-1999-2439),<br>
Samuel Leweke<sup>1</sup> (ORCID: 0000-0001-9471-4511),<br>
Johannes Schmölder<sup>1</sup> (ORCID: 0000-0003-0446-7209),<br>
Gregor Gassner<sup>2,3</sup> (ORCID: 0000-0002-1752-1158),<br>
Eric von Lieres<sup>1</sup> (ORCID: 0000-0002-0309-8408)<br>

<sup>1</sup> Forschungszentrum Jülich, IBG-1: Biotechnology, Jülich, Germany<br>
<sup>2</sup> Department of Mathematics and Computer Science, University of Cologne, Cologne, Germany<br>
<sup>3</sup> Center for Data and Simulation Science, University of Cologne, Cologne, Germany

# Description
Supplement code to recreate evaluation.

## License
We note that CADET is distributed under a different license than the one given here, which solely refers to the evaluation scripts.

## General procedure:
* Either clone DG branch of CADET [here](https://github.com/modsim/CADET/tree/feature/dg) or use the CADET-DG commit in this repo (to use the exact same version that was used in the paper) and install following the [installation guide](https://cadet.github.io/master/getting_started/installation.html). Note that in the future, after we merged the DG extension into the main branch, this step becomes obsolete and CADET can be installed via conda.
* Create conda environment using the environment.yml file.
* Recreate evaluation by executing the code given in eval_CADET_DG.py according to the instructions given therein.

## Requirements

### Software:
- [CADET (DG branch)](https://github.com/modsim/CADET/tree/feature/dg)
- Anaconda, Python = 3.9
- Git

### Anaconda

1. Install Python Environment manager with Python 3.9 (Anaconda with Anaconda Navigator).
2. Create new environment using the environment.yml by executing ``conda env create -f environment.yml``.
