### Signal Dependant 3D Point Cloud Compression

Use Python 3.10 to avoid pybind11 conflicts with Python >=3.11
Install Python Run Length Golomb Rice encoder via:

```bash
git submodule update --init --recursive
cd PyRLGR
pip install .
```

Ensure you have needed packages to build
Sometimes the PyRLGR build can fail because it doesn't find the "Python.h" in that case just do:

# Arch
```bash
sudo pacman -S cmake python-setuptools base-devel
sudo pacman -Syu cmake
```

# Ubuntu
```bash
sudo apt-get install cmake python-setuptools python3-dev
```

Install Point Cloud Attribute Dependant Compression (PCADC) and needed libraries
```bash
pip install .
pip install -r requirements.txt
```


