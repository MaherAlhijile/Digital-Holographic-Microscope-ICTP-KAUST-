
# Digital Holography Microscope (DHM) for Automatic Disease Identification Using AI
# Collection of Resources and Findings

# System Versions:
## Local Version
This is a standalone desktop software designed to work with a DHM (Digital Holographic Microscopy) add-on for regular microscopes. It handles live acquisition, phase reconstruction, ROI selection, and image processing locally using the connected Basler camera and the Pylon library. Can be found under the local directory  

## Before starting
* Download and install pylon library, give sudo permit to install USB rules, in case of doubt, follow `Install.txt` instructions.

[Install link, click here:](https://www.baslerweb.com/en/downloads/software/3032421996/?downloadCategory.values.label.data=pylon)

* Setup a .venv in the Top-level of the local repository.

```Bash
cd <git_address>
python3 -m venv .venv
```
or create it using vscode environemnts.

Activate the .venv and install the requirements library
```Bash
source .venv/bin/activate
pip install -r venv_requirements/requirements.txt
```

*To run the software

```Bash
chmod +x Run.sh #Only the first time
source Run.sh
```
### Local Version Enhancements:
  * Organized the program to follow a logical sequence of operations
  * Added functionality to perform all operations with a single click
  * Resolved issues with negative/positive cells in phase computation
  * Enabled batch processing of multiple images
  * Fixed ROI selection – now works correctly from the first click
  * Corrected the thickness functions – each function now opens in a separate, * independent window

## Remote Version
This is a web application designed to work with our in-house designed DHM system. It supports remote image upload, cloud-based reconstruction, AI-powered diagnostics, and batch analysis. It is built for scalability and accessibility, enabling point-of-care usage without the need for local processing power. Can be found under the remote directory  

### Version specifications:
  * Improved the noise reduction functionality using Artificial Intelligence
  * Trained a machine learning model for remote, label-free, automated point-of-care disease diagnosis
  * Used features extracted from DHM to train ML algorithms for detecting diseases such as malaria and sickle cell anemia


