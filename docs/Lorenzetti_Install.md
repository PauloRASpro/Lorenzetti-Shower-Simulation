# Lorenzetti Showers — Setup & Run on Google Cloud (Ubuntu 22.04 LTS)


End-to-end pipeline to prepare a VM on Google Cloud (Ubuntu 22.04 LTS), install system
dependencies, set up Docker, obtain the Lorenzetti image, build the project, and run the
simulation chain (EVT → HIT → ESD → AOD).

Summary: Python venv → system dependencies → Docker → image lorenzetti/lorenzetti:latest → build → simulations.

---

## TABLE OF CONTENTS


1. Requirements
2. Python environment (optional)
3. System dependencies
4. Docker (official install)
5. Pull Lorenzetti image + build
6. Run simulations
7. License
8. References
---

## 1. REQUIREMENTS

- Linux VM with Ubuntu 22.04 LTS (tested on Google Compute Engine).
- User with sudo privileges.
- SSH enabled to copy files off the VM if needed.
- ~100 GB disk recommended (ROOT, Geant4, artifacts and outputs).
---

## 2. PYTHON ENVIRONMENT

### 2.1. Check Python 3

`python3 --version`

### 2.2. Create a virtual environment (e.g., ~/.venvs/base)

`python3 -m venv ~/.venvs/base`

### 2.3. Activate the venv

`source ~/.venvs/base/bin/activate`

### 2.4. Upgrade installers

`python -m pip install --upgrade pip setuptools wheel`

### 2.5. Useful Python packages

`python -m pip install   numpy pandas scikit-learn seaborn jupyterlab tqdm atlas-mpl-style   twine pyhepmc prettytable expand_folders rich_argparse loguru`

OBS.: How to reactivate later:
- Open a new terminal and run:
  
`source ~/.venvs/base/bin/activate`

---

## 3. SYSTEM DEPENDENCIES

`sudo apt-get update`

`sudo apt-get install -y   autoconf automake libtool pkg-config   libseccomp-dev zlib1g-dev   squashfs-tools squashfs-tools-ng   fuse fuse2fs libfuse-dev   uidmap runc   cryptsetup   git wget`

`sudo apt update`

`sudo apt -y upgrade`

`sudo apt install -y build-essential git cmake ninja-build curl bzip2 ca-certificates pkg-config`

`sudo apt update && sudo apt install -y python3-full python3-venv python3-dev build-essential gfortran libatlas-base-dev`


(optional) Python packages via pip — if you did not use the venv above

`pip install --upgrade pip setuptools wheel`

`pip install numpy pandas scikit-learn seaborn jupyterlab tqdm atlas-mpl-style twine pyhepmc prettytable expand_folders rich_argparse loguru`

---

## 4. DOCKER (OFFICIAL INSTALL)

### 4.1. remove old versions (if any)

`sudo apt update`

`sudo apt-get remove -y docker docker.io docker-engine containerd runc || true`


### 4.2. deps + GPG key

`sudo apt-get update`

`sudo apt-get install -y ca-certificates curl gnupg`

`sudo install -m 0755 -d /etc/apt/keyrings`

`curl -fsSL https://download.docker.com/linux/$(. /etc/os-release; echo "$ID")/gpg   | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg`

`sudo chmod a+r /etc/apt/keyrings/docker.gpg`

### 4.3. official apt repo

`echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg]   https://download.docker.com/linux/$(. /etc/os-release; echo "$ID")   $(. /etc/os-release; echo "$VERSION_CODENAME") stable"   | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null`

### 4.4. install

`sudo apt-get update`

`sudo apt-get install -y docker-ce docker-ce-cli containerd.io   docker-buildx-plugin docker-compose-plugin`


### 4.5. enable and start

`sudo systemctl enable --now docker`

### 4.6. use without sudo (optional; re-open your session afterwards)

`sudo usermod -aG docker $USER`

`newgrp docker`

### 4.7. test

`docker run --rm hello-world`

---

## 5. PULL LORENZETTI IMAGE + BUILD

Persistence tip: mount a host directory as a volume to store outputs (/data on host → /work in the container).
This ensures your results survive container restarts.

### 5.1. Pull the image

`docker pull lorenzetti/lorenzetti:latest`

### 5.2. List local images

`docker images`

### 5.3. Create a working directory on the host (to persist outputs)

`mkdir -p ~/lorenzetti_work`

`cd ~/lorenzetti_work`

### 5.4. Start an interactive shell with the volume mounted

`docker run --rm -it -v "$PWD":/work -w /work lorenzetti/lorenzetti:latest bash`

Inside the container:

### 5.5. Clone and build the repository

`git clone https://github.com/lorenzetti-hep/lorenzetti.git && cd lorenzetti`

`make`

`rm -rf build/lib/`

`source build/lzt_setup.sh`

### 5.6. Export paths (ensure the build is on PATH/LD)

`export PYTHONPATH=$(pwd)/build/python:$PYTHONPATH`

`export LD_LIBRARY_PATH=$(pwd)/build:$LD_LIBRARY_PATH`

`export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`

Quick check

`python3 -c "import filters"`

---

## 6. RUN THE SIMULATIONS (Example)

In this section, we present an example simulation using an electron gun, where the transverse energy spans from 5 to 500 GeV and the cluster pseudorapidity (η) ranges from −2.47 to 2.47. To avoid known detector non-uniformities, the calorimeter crack regions are excluded—specifically, −1.52 to −1.37 and 1.37 to 1.52 in η—so that the generated events sample only the well-instrumented barrel and endcap acceptance.

The standard sequence is EVT → HIT → ESD → AOD.

### 6.1. Event generation (EVT)

Electron gun over eta ranges (excluding the 1.37–1.52 crack)


`python build/scripts/filters/gen_single.py   --particle Electron   --nov 100   --output-file EVT_endcap_neg.root   --energy-min 5 --energy-max 200   --eta-min -2.47 --eta-max -1.52   --seed 1001`

`python build/scripts/filters/gen_single.py   --particle Electron   --nov 100   --output-file EVT_barrel.root   --energy-min 5 --energy-max 200   --eta-min -1.37 --eta-max 1.37   --seed 1002`

`python build/scripts/filters/gen_single.py   --particle Electron   --nov 100   --output-file EVT_endcap_pos.root   --energy-min 5 --energy-max 200   --eta-min 1.52 --eta-max 2.47   --seed 1003`

Merge into a single EVT

`hadd -f EVT.root      EVT_barrel.root      EVT_endcap_neg.root      EVT_endcap_pos.root`

### 6.2. Hotfix before propagation (HIT)

Backup the script

`cp build/scripts/reco/simu_trf.py build/scripts/reco/simu_trf.py.bak`

Patch: ensure input_file/output_file are not lists

`sed -i 's|args.input_file = Path(args.input_file)|    # --- begin hotfix: argparse may give a list even for single -i ---    if isinstance(args.input_file, list):        if len(args.input_file) != 1:            raise SystemExit("Provide exactly one -i INPUT_FILE");        args.input_file = args.input_file[0]    args.input_file = Path(args.input_file)|' build/scripts/reco/simu_trf.py`

`sed -i 's|args.output_file = Path(args.output_file)|    if isinstance(args.output_file, list):        if len(args.output_file) != 1:            raise SystemExit("Provide exactly one -o OUTPUT_FILE");        args.output_file = args.output_file[0]    args.output_file = Path(args.output_file)|' build/scripts/reco/simu_trf.py`

Ensure the simulation env (often set in the image already):

`[ -f /opt/root/bin/thisroot.sh ] && source /opt/root/bin/thisroot.sh
[ -f /opt/geant4/bin/geant4.sh ] && source /opt/geant4/bin/geant4.sh`

### 6.3. Shower propagation (HIT)


`simu_trf.py   --enable-magnetic-field   -i EVT.root   -o HIT.root   --nov 100   -nt 8   --overwrite`

### 6.4. Digitization (ESD)


`digit_trf.py -i HIT.root -o ESD.root`

Alternative with cross-talk and COF:

`digit_trf.py -i HIT.root -o ESD.root --simulateCrossTalk --estimationMethodHAD COF`

### 6.5. Reconstruction (AOD)


`reco_trf.py -i ESD.root -o AOD.root`

Where are the output files?
- If you ran with -v "$PWD":/work -w /work, all outputs (EVT.root, HIT.root, ESD.root, AOD.root, etc.)
  will be in the host directory where you started docker run.

---

## 7. LICENSE

Lorenzetti software follows the official repository's license.

---

## 8. REFERENCES

- Official repository: https://github.com/lorenzetti-hep/lorenzetti
- Docker image: lorenzetti/lorenzetti:latest on Docker Hub

---
