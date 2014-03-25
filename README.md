# FiVoNAGI

Name : Finite Volume Nonlinear Acoustics GPU Implementation (FiVoNAGI)

Authors : Roberto Velasco Segura and Pablo L. Rendón

Registration numbers: 03-2014-022811451900-01 INDAUTOR México

Source repository : https://github.com/rvelseg/FiVoNAGI

## Description

This code calculates an approximate solution for nonlinear and
dissipative acoustic field, using conservation laws obtained with all
but one hypotheses used to obtain Westervelt equation.

The code is implemented in 2D using a finite volume method. It
uses 2D texture memory and OpenGL display.

## Papers

First results of this code have been published as preprint in
http://arxiv.org/abs/1311.3004 , if you use this code in your research
please cite using

> @article{velasco2013finite,
> 	title={A finite volume approach for the simulation of nonlinear
> 	dissipative acoustic wave propagation},
> 	author={Velasco-Segura, Roberto and Rend{\'o}n, Pablo L},
> 	journal={arXiv preprint arXiv:1311.3004},
> 	year={2013}
> 	}

## License

See license.txt in the root directory of the repository.

## System requirements

This code has been tested in the following system, we are not sure if
it runs, and how it performs, in different systems:

Hardware

* core i3 processor 
* 16 GB RAM
* 500 GB HDD (dedicated to these simulations)
* 2.0 GPU CUDA compute capability
* 6 GB GPU RAM
* 448 CUDA cores

Software

* Linux Debian/Sid
* GNU bash 4.2.45
* GNU Make 3.81
* CUDA 5.5.0
* g++ 4.8.2
* python 2.7.6
* gnuplot 4.6
* pdfTeX 3.1415926-2.5-1.40.14

Any feedback on performance over different systems is welcome.

## Usage

First, clone or download (and decompress) the repository.

To execute the existing code:

* `cd` to the root directory of the repository
* if you want to see some action before more reading try `make APP=hifu-beam display`
* execute `make help` and see further instructions

To create new applications:

* make a new directory inside `apps` directory
* replicate the structure of existing applications using code for your
specific system

## Structure

The core of this code is in the `common` directory. Applications are
directories in any level within the `apps` directory, inside each application
directory should be files defining those qualities that make the
corresponding application different from the other applications,
e.g. boundary conditions, Riemann solver or initial values. If two
applications share something, it is recommended to create a third
directory to place common files instead of repeating them, as is the
case of `apps/na`.

As you have probably already noticed, the structure of this code
reproduces a few parts of the structure of CLAWPACK

> R. J. LeVeque, M. J. Berger, et. al., Clawpack Software 4.6.1,
> www.clawpack.org, last visited: november 2013.

we have learned a lot looking at that code. Indeed some of this code
replicates, with the context needs, parts of CLAWPACK code. To
facilitate comparison indications like

> For comparison with CLAWPACK see: path/to/some/clawpack/file.f

have been placed as comments in the corresponding files. Integration
would be highly desirable for us.
