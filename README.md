# Robust Discrete Differential Operators for Wild Geometry

This repository contains the source code of the paper "Robust Discrete Differential Operators for Wild Geometry" by Wagner and Botsch.

## Installing, Building and Running the Demos

### Prerequisites

Please clone this repository recursively, as it uses submodules:

    git clone --recursive git@github.com:sdwagner/wildDDG.git

While this repository includes nearly all geometry processing libraries as submodules, [CGAL](https://www.cgal.org/download.html) needs
to be installed manually via, e.g., vcpkg, apt-get, or homebrew. 
This should already include installations of MPFR, GMP, and Boost, which are also necessary to build this program.

### Building the Code

Building the program is then just:
    
    cd wildDDG && mkdir build && cmake .. && make -j

This should build all the necessary targets and dependencies. 


### Running the Code

Finally, you can start the demo by running

    ./surface_viewer

If you want to use a specific mesh, you can just drag and drop it into the window.

The different benchmarks can be run using the following commands
    
    ./tri_tests
    ./unstructured_tri_tests
    ./poly_tests

Running the benchmark on Thingi10k first requires you to follow the instructions in `scripts/filter_thingi10k.ipynb`, after
which you can start the benchmark with

    ./thingi10k_tests



## BibTeX

    @inproceedings{wagner2025robust,
        author = {Wagner, Sven D. and Botsch, Mario},
        title = {{Robust Discrete Differential Operators for Wild Geometry}},
        booktitle = {Vision, Modeling, and Visualization},
        editor = {Egger, Bernhard and G{\"u}nther, Tobias},
        year = {2025},
        publisher = {The Eurographics Association}
    }
