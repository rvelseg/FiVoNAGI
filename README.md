# FiVoNAGI

Name : Finite Volume Nonlinear Acoustics GPU Implementation (FiVoNAGI)

Authors : Roberto Velasco Segura and Pablo L. Rendón

Affiliations : Grupo de Acústica y Vibraciones, Centro de Ciencias
Aplicadas y Desarrollo Tecnológico (CCADET), Posgrado en Ciencias
Físicas (PCF), Universidad Nacional Autónoma de México (UNAM).

Registration numbers: 03-2014-022811451900-01 INDAUTOR México

Funding: PAPIIT IN109214.

Source repository : https://github.com/rvelseg/FiVoNAGI

## Description

This code calculates an approximate solution for nonlinear and
dissipative acoustic field, using conservation laws obtained with all
but one hypotheses used to obtain Westervelt equation.

The code is implemented in 2D using a finite volume method. It
uses 2D texture memory and OpenGL display.

## Papers

First results of this code have been published in "Wave Motion"
journal.

> R. Velasco-Segura, P. L. Rendón. A finite volume approach for the
> simulation of nonlinear dissipative acoustic wave
> propagation. 2015. Wave Motion.
> http://dx.doi.org/10.1016/j.wavemoti.2015.05.006

## License

See license.txt in the root directory of the repository.

## System requirements

This code has been tested in the following system, we are not sure if
it runs, and how it performs, in different systems:

Hardware

* core i5 processor
* 16 GB RAM
* 500 GB HDD (dedicated to these simulations)
* 2.0 GPU CUDA compute capability
* 6 GB GPU RAM
* 448 CUDA cores

Software

* Linux Debian/Jessie
* GNU bash 4.3.30
* GNU Make 4.0
* CUDA 6.0.1
* g++ 4.9.1
* python 2.7.8
* gnuplot 4.6
* pdfTeX 3.1415926-2.6-1.40.15

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
with this or other FOSS project would be highly desirable for us.
