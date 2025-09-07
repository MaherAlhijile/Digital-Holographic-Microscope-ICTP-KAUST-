# Digital Holography Microscope (DHM) for Automatic Disease Identification Using AI

This repository contains two implementations of a Digital Holography Microscope (DHM) Software:  
1. **Local Version** – standalone desktop software designed to work with the DHM add-on for regular microscopes. It handles live acquisition, phase reconstruction, ROI selection, and image processing locally using the connected Basler camera and the Pylon library. Can be found under the local directory  

2. **Remote Version** – This is a web application designed to work with our in-house designed DHM system which is built with FastAPI (backend) and HTML/JS (frontend). It is designed for point-of-care and cloud-based diagnostics.. It supports remote image upload, cloud-based reconstruction, AI-powered diagnostics, and batch analysis. It is built for scalability and accessibility, enabling point-of-care usage without the need for local processing power. 

---

## Prerequisites
Before starting, make sure you have the following installed on your system:

- an operating system installed on the Raspberry Pi
- Python **3.0+**
- **pip** (Python package manager)
- **Git** (to clone the repository)
- an IDE 

Start by typing the following command in the terminal:
 pip install -r requirements.txt
Which will install all dependencies for both local and remote (front & back ends) versions

Then download and install pylon library from https://www.baslerweb.com/en/downloads/software/2907135243/ , give sudo permit to install USB rules, in case of doubt, follow `Install.txt` instructions.

---

## Local Version
This is a standalone desktop software designed to work with a DHM (Digital Holographic Microscopy) add-on for regular microscopes.  
It handles live acquisition, phase reconstruction, ROI selection, and image processing locally using the connected Basler camera and the Pylon library.  

Code can be found under the **`/local`** directory.

### Setup
1. Install the Pylon library (follow `Install.txt` if needed). Make sure to give `sudo` permission to install USB rules.
2. Create a virtual environment in the top-level of the local repository:
   ```bash
   cd <git_address>
   python3 -m venv .venv

