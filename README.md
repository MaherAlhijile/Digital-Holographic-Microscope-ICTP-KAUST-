# Digital Holography Microscope (DHM) for Automatic Disease Identification Using AI

This repository contains two implementations of a Digital Holography Microscope (DHM) Software:  
1. **Local Version** – standalone desktop software designed to work with the DHM add-on for regular microscopes. It handles live acquisition, phase reconstruction, ROI selection, and image processing locally using the connected Basler camera and the Pylon library. Can be found under the local directory  

2. **Remote Version** – This is a web application designed to work with our in-house designed DHM system which is built with FastAPI (backend) and HTML/JS (frontend). It is designed for point-of-care and cloud-based diagnostics.. It supports remote image upload, cloud-based reconstruction, AI-powered diagnostics, and batch analysis. It is built for scalability and accessibility, enabling point-of-care usage without the need for local processing power. 

---

## Prerequisites

If using the remote version, make sure to Setup a virtual envoriment in the Top-level of the local repository.
start by typing the command
```bash
python -m venv .venv
````
then activate the envorimanet using
```bash
source .venv/bin/activate
```
This will ensure no conflict will happen and the software will run smoothly

Then download and install pylon library from https://www.baslerweb.com/en/downloads/software/2907135243/ , give sudo permit to install USB rules, in case of doubt, follow `Install.txt` instructions.

Finally, we install the motors depenccies. in the server side (must be linux) install the WiringPi library for controling the motors
first clone it
```bash
git clone https://github.com/WiringPi/WiringPi.git
```
then enter the directory
```bash
cd WiringPi
```
and build the libaray
```bash
./build
```
now check if it was successfully installed
```bash
gpio -v
```

Now we have to install the python wrapper, clone the following repo

```bash
git clone https://github.com/sultanf110/stepper_motor.git
```
then enter the dirctory and install
```bash
pip install .
```

For both versions, make sure you have the following installed on your system:
- an operating system installed on the Raspberry Pi
- Python **3.0+**
- **pip** (Python package manager)
- **Git** (to clone the repository)
- an IDE 

Start by typing the following command in the terminal:
```bash
pip install -r requirements.txt
```
Which will install all dependencies for both local and remote (front & back ends) versions


---

## Local Version
This is a standalone desktop software designed to work with a DHM (Digital Holographic Microscopy) add-on for regular microscopes.  
It handles live acquisition, phase reconstruction, ROI selection, and image processing locally using the connected Basler camera and the Pylon library.  

Code can be found under the **`/local`** directory.

### Local Version Enhancements:
  * Organized the program to follow a logical sequence of operations
  * Added functionality to perform all operations with a single click
  * Resolved issues with negative/positive cells in phase computation
  * Enabled batch processing of multiple images
  * Fixed ROI selection – now works correctly from the first click
  * Corrected the thickness functions – each function now opens in a separate, * independent window


## Remote Version
This is a web application designed to work with our in-house designed DHM system which is built with FastAPI (backend) and HTML/JS (frontend). It is designed for point-of-care and cloud-based diagnostics.. It supports remote image upload, cloud-based reconstruction, AI-powered diagnostics, and batch analysis. It is built for scalability and accessibility, enabling point-of-care usage without the need for local processing power.

Features include:

* Remote image upload and analysis

* 3D and 1D thickness profiling

* Live camera integration (via Basler Pylon)

* Configurable microscope parameters via frontend GUI

### Version specifications:
  * Improved the noise reduction functionality using Artificial Intelligence
  * Trained a machine learning model for remote, label-free, automated point-of-care disease diagnosis
  * Used features extracted from DHM to train ML algorithms for detecting diseases such as malaria and sickle cell anemia
  * REST API endpoints for:
       ** /run_phase_difference – Compute phase maps
       
       ** /compute_3d – Generate 3D surface thickness data
       
       ** /compute_1d – Extract 1D thickness profile
       
       ** /check_spectrum – Fourier spectrum validation
       
       ** /start_camera, /stop_camera, /camera_feed – Camera control & streaming


### Folder Structure:
```
/remote
   ├── backend/
   │     ├── server.py         # FastAPI server + endpoints
   │     ├── sys_functions.py  # Phase reconstruction + analysis
   ├── frontend/
   │     ├── index.html        # Web GUI
   │     ├── css/
   │     ├── js/
```
### How to Run:
1. Navigate to the backend folder
```Bash
cd remote/backend
uvicorn --DHM 192.168.1.121 server:app --port:8000 --reload
```

This starts the FastAPI server at http://192.168.1.121.

2. Open the frontend (remote/frontend/index.html) in a browser.

3. Configure microscope/connection parameters

4. Upload object & reference images or connect to the Basler camera

5. Run computations (Phase Difference, ROI, 3D profile, 1D profile)

