#################
# Variables
#################

# to be able to use 'tee' command
SHELL:=/bin/bash

# The absolute path to the root directory of FiVoNAGI.
ROOT = $(shell pwd)

# These simulations typically have hundreds of GB as output, use the
# following variable to make the results be deployed in a unit with
# enough free space. Absolute path with no quotes, and no trailing
# slash. Default is: ${ROOT}/results
RESULTS := ${ROOT}/results

ifdef APP
# Source files of the current APP
APP_SRC_H = $(shell ls apps/${APP}/*.h)
APP_SRC_CU = $(shell ls apps/${APP}/*.cu)
endif

#################
# Help
#################

# To get some information on how this file works use 
#
# make help

help : greetings list_apps
	@echo "To compile 'appname' use 'make APP=appname compile', note that 'appname' can contain one or more slashes '/' in it."
	@echo "To run the configured simulation for 'appname', without saving data do disk, use 'make APP=appname display' "
	@echo "To run the configured simulation for 'appname', saving data do disk, use 'make APP=appname data' "
	@echo "To make the configured plots for 'appname' use 'make APP=appname plots' "
	@echo "To make the configured videos for 'appname' use 'make APP=appname videos' "
	@echo "To run the simulation, save the data, make the configured plots and videos for 'appname' use 'make APP=appname all' "
	@echo "Executions will try not to overwrite data, plots or videos, if you want to do it use 'make APP=appname clean_data data' or 'make APP=appname clean_videos videos' or similar."
	@echo "To interrupt an execution use Ctrl+C"

#################
# Targets
#################

apps/%/FiVoNAGI : $(APP_SRC_H) $(APP_SRC_CU)
	@echo "The following compilation is not for real simulations, it is just for you to test the compilation process. Variables passed with -D are the simplest possible options, maybe not what you need. Configure those variables in the script run.sh inside the application directory and then perform real simulations with 'make APP=appname display' or 'make APP=appname data', a new compilation will be performed for every simulation anyway."
	cd apps/$*/ && nvcc -D ROOT=${ROOT} -D DEPLOY="$(RESULTS)" -D PRECISION=1 -D CFLW=0.99 -D ETAC=1 -o FiVoNAGI -lcuda -lcudart  -lm -lGL -lGLU -lglut -lpthread -arch=sm_13 ./driver.cu

$(RESULTS)/%/data/flag : 
	( [ -d $(RESULTS)/$*/data/ ] && echo "Deploy directory found" ) || mkdir -p $(RESULTS)/$*/data
	touch $(RESULTS)/$*/data/stdout.log
	touch $(RESULTS)/$*/data/stderr.log
	cd apps/$*/ && ./run.sh "${ROOT}" "$(RESULTS)/$*/data" "EXPORT" > >(tee $(RESULTS)/$*/data/stdout.log) 2> >(tee $(RESULTS)/$*/data/stderr.log >&2)
	@echo "data deploy path: $(RESULTS)/$*/data/"
	touch $(RESULTS)/$*/data/flag

$(RESULTS)/%/plots/flag : 
	[ -d $(RESULTS)/$*/data/ ]
	( [ -d $(RESULTS)/$*/plots/ ] && echo "Deploy directory found" ) || mkdir -p $(RESULTS)/$*/plots
	touch $(RESULTS)/$*/plots/stdout.log
	touch $(RESULTS)/$*/plots/stderr.log
	cd apps/$*/ && ./plots.sh "$(RESULTS)/$*/data" "$(RESULTS)/$*/plots" > >(tee $(RESULTS)/$*/plots/stdout.log) 2> >(tee $(RESULTS)/$*/plots/stderr.log >&2)
	@echo "plots deploy path: $(RESULTS)/$*/plots/"
	touch $(RESULTS)/$*/plots/flag

$(RESULTS)/%/videos/flag : 
	[ -d $(RESULTS)/$*/data/ ]
	( [ -d $(RESULTS)/$*/videos/ ] && echo "Deploy directory found" ) || mkdir -p $(RESULTS)/$*/videos
	touch $(RESULTS)/$*/videos/stdout.log
	touch $(RESULTS)/$*/videos/stderr.log
	cd apps/$*/ && ./videos.sh "$(RESULTS)/$*/data" "$(RESULTS)/$*/videos" > >(tee $(RESULTS)/$*/videos/stdout.log) 2> >(tee $(RESULTS)/$*/videos/stderr.log >&2)
	@echo "videos deploy path: $(RESULTS)/$*/videos/"
	touch $(RESULTS)/$*/videos/flag

#################
# Phony targets
#################

.PHONY : greetings help list_apps \
check compile display data plots videos all \
clean_compile clean_data clean_plots clean_videos clean_all

check : 
ifdef APP
	@echo "looking for ./apps/${APP}/driver.cu"
ifeq ($(wildcard apps/${APP}/driver.cu),) 
	$(error APP not found. See 'make help')
else 
	@echo "APP ${APP} found."
endif
else
	$(error APP not defined. See 'make help')
endif

compile : check apps/${APP}/FiVoNAGI

display : check
	cd apps/${APP}/ && ./run.sh "${ROOT}" "."

data : check $(RESULTS)/${APP}/data/flag

plots : check $(RESULTS)/${APP}/plots/flag

videos : check $(RESULTS)/${APP}/videos/flag

all : check data plots videos

clean_compile : check
	-rm apps/${APP}/FiVoNAGI

clean_data : check
	-rm -rf $(RESULTS)/${APP}/data

clean_plots : check
	-rm -rf $(RESULTS)/${APP}/plots

clean_videos : check
	-rm -rf $(RESULTS)/${APP}/videos

clean_all : check clean_compile clean_data clean_plots clean_videos

list_apps : 
	@echo "Following are the available applications:"
	@find . -name "driver.cu" | sed 's|/driver.cu||' | sed 's|^./apps/||'

greetings : 
	@echo "This is FiVoNAGI Makefile running on"
	@echo "${ROOT}"
