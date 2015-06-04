#!/bin/bash
#

# For comparison with CLAWPACK see: clawpack-4.6.1/apps/acoustics/2d/example1/setplot.py

as=(30 20 10)
etas=(20 41 82)
cfls=(60 70 80 90 99)
ps=(1 2)

RESULTS_PATH=$1
DEPLOY_PATH=$2
APP_PATH=`pwd`

GNUPLOT_VARS="resultsPath='${RESULTS_PATH}'; deployPath='${DEPLOY_PATH}'; appPath='${APP_PATH}'"

#######################################################
# rearrange some data before plotting

std_cfl_path="${DEPLOY_PATH}/dr_std_cfl"
if [ ! -d "$std_cfl_path" ]
then
    mkdir $std_cfl_path

    for pi in "${ps[@]}"
    do
	for etai in "${etas[@]}"
	do
	    for cfli in "${cfls[@]}"
	    do
		compname="cfl_std-eta${etai}-cfl${cfli}-p${pi}"
		filecomp="${std_cfl_path}/${compname}.dat"
		if [ -f "$filecomp" ]
		then
		    rm "$filecomp"
		fi
		touch "$filecomp"
		echo "# a n T frame cfl std" >> $filecomp
		for ai in "${as[@]}" 
		do
		    execname="eta${etai}-cfl${cfli}-p${pi}-a${ai}"
		    dir="d-$execname"
		    file="${RESULTS_PATH}/$dir/u1-cfl.dat"
		    if [ -f $file ]
		    then
			line=`tail -n 1 $file`
			echo "${ai} $line" >> $filecomp
		    fi
		done
	    done
	done
    done
fi

########################################################################

# This looks more like a makefile, echoing every executed command,
# maybe this file should be some kind of makefile.

cmd="./plots/error_collect.py \"${RESULTS_PATH}\" \"${DEPLOY_PATH}\""
echo $cmd
eval $cmd
cmd="./plots/error_as.py \"${RESULTS_PATH}\" \"${DEPLOY_PATH}\""
echo $cmd
eval $cmd
cmd="./plots/error_etas.py \"${RESULTS_PATH}\" \"${DEPLOY_PATH}\""
echo $cmd
eval $cmd
cmd="./plots/error_table_1.py \"${RESULTS_PATH}\" \"${DEPLOY_PATH}\""
echo $cmd
eval $cmd
cmd="./plots/error_table_2.py \"${RESULTS_PATH}\" \"${DEPLOY_PATH}\""
echo $cmd
eval $cmd
cmd="./plots/error_table_3.py \"${RESULTS_PATH}\" \"${DEPLOY_PATH}\""
echo $cmd
eval $cmd
cmd="./plots/error_table_4.py \"${RESULTS_PATH}\" \"${DEPLOY_PATH}\""
echo $cmd
eval $cmd
cmd="./plots/error_cfls.py \"${RESULTS_PATH}\" \"${DEPLOY_PATH}\""
echo $cmd
eval $cmd

cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/error_as.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/error_etas.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/error_etas_special_1.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/std_cfl.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/cfls.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/etas.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/ps.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/single_execution.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/tot_a.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/tot_a_special_1.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/typical.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/error_cfls.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./spectrum/spectrum_special_1.gp"
echo $cmd
eval $cmd

# gnuplot -e "$GNUPLOT_VARS" spectrum_se.gp
# gnuplot -e "$GNUPLOT_VARS" spectrum_etas.gp
# gnuplot -e "$GNUPLOT_VARS" spectrum_special_1.gp

# copy the figures to be used in the paper to a single directory
export_path="${DEPLOY_PATH}/plots_export_1"
if [ ! -d "$export_path" ]
then
    mkdir $export_path

    cp -a "${DEPLOY_PATH}/plots_spectrum_special_1/hb_spectrum_special_1_sa-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_std_cfl/hb_cfl_std-2.png" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_tot_a_special_1/hb_a30-p2-sa-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_typical/hb_typical_fix-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_error_etas_special_1/hb_error-etas-p2-L1-sa-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/table_error_4/conv_rates_stats_as-p2.dat" "${export_path}"
fi

# figures for the thesis
export_path="${DEPLOY_PATH}/plots_export_2"
if [ ! -d "$export_path" ]
then
    mkdir $export_path

    cp -a "${DEPLOY_PATH}/plots_spectrum_special_1/hb_spectrum_special_1_sa-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_std_cfl/hb_cfl_std-2.png" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_tot_a_special_1/hb_a10-p2-sa-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_tot_a_special_1/hb_a20-p2-sa-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_tot_a_special_1/hb_a30-p2-sa-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_typical/hb_typical_fix-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_error_etas_special_1/hb_error-etas-p2-L1-sa-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/table_error_4/conv_rates_stats_as-p2.dat" "${export_path}"
fi
