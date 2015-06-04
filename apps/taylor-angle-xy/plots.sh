#!/bin/bash
#

# For comparison with CLAWPACK see: clawpack-4.6.1/apps/acoustics/2d/example1/setplot.py

ps=(1 2)
etas=(5 10 20 41 82)
angles=(    "0"     "PI/32"     "PI/16"     "PI/8"     "PI/4") 
angles_str=("00000" "PIo32"     "PIo16"     "PIo08"    "PIo04")
cfls=(60 70 80 90 99)

RESULTS_PATH=$1
DEPLOY_PATH=$2
APP_PATH=`pwd`

GNUPLOT_VARS="resultsPath='${RESULTS_PATH}'; deployPath='${DEPLOY_PATH}'; appPath='${APP_PATH}'"

# TODO: avoid data rearrangement using awk inside gnuplot to get
# desired lines, it may be necessary the use of get_data as well
# http://stackoverflow.com/questions/18583180

#######################################################
# rearrage some data before plotting

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
		echo "# angle n T frame cfl std" >> $filecomp
		anglek=0
		for anglei in "${angles[@]}" 
		do
		    execname="eta${etai}-cfl${cfli}-p${pi}-angle${angles_str[anglek]}"
		    dir="${RESULTS_PATH}/d-$execname" 
		    file="$dir/u1-cfl.dat"
		    if [ -f $file ]
		    then
			anglelow="$(echo $anglei | tr '[A-Z]' '[a-z]')"
			anglen=`python -c "from numpy import * ; print $anglelow"`
			line=`tail -n 1 $file`
			echo "$anglen $line" >> $filecomp
		    fi
		    anglek=$[anglek + 1]
		done
	    done
	done
    done
fi

###########################################################
# rearrage some more data before plotting

error_path="${DEPLOY_PATH}/dr_error_angles"
if [ ! -d "$error_path" ]
then
    mkdir $error_path

    for pi in "${ps[@]}"
    do
	lsk=0
	for etai in "${etas[@]}"
	do
	    cflk=0
	    for cfli in "${cfls[@]}"
	    do
		compname="error-eta${etai}-cfl${cfli}-p${pi}"
		filecomp="${error_path}/${compname}.dat"
		if [ -f "$filecomp" ]
		then
		    rm "$filecomp"
		fi
		touch "$filecomp"
		echo "# angle frame T error_L1 error_L2 error_Li" >> $filecomp
		anglek=0
		for anglei in "${angles[@]}" 
		do
		    execname="eta${etai}-cfl${cfli}-p${pi}-angle${angles_str[anglek]}"
		    dir="${RESULTS_PATH}/d-$execname"
		    file="$dir/u1-error.dat"
		    if [ -f $file ]
		    then
			anglelow="$(echo $anglei | tr '[A-Z]' '[a-z]')"
			anglen=`python -c "from numpy import * ; print $anglelow"`
			line=`tail -n 1 $file`
			echo "$anglen $line" >> $filecomp
		    fi
		    anglek=$[anglek + 1]
		done
		cflk=$[cflk + 1]
		lsk=$[lsk + 1]
	    done
	done
    done
fi

######################################################
# rearrage even more data before plotting

error_etas_path="${DEPLOY_PATH}/dr_error_etas_GPU"
if [ ! -d "$error_etas_path" ]
then
    mkdir $error_etas_path

    for pi in "${ps[@]}"
    do
	lsk=0
	anglek=0
	for anglei in "${angles[@]}" 
	do
	    cflk=0
	    for cfli in "${cfls[@]}"
	    do
		compname="error-cfl${cfli}-p${pi}-angle${angles_str[anglek]}"
		filecomp="${error_etas_path}/${compname}.dat"
		if [ -f "$filecomp" ]
		then
		    rm "$filecomp"
		fi
		touch "$filecomp"
		echo "# eta frame T error_L1 error_L2 error_Li" >> $filecomp
		for etai in "${etas[@]}"
		do
		    execname="eta${etai}-cfl${cfli}-p${pi}-angle${angles_str[anglek]}"
		    dir="${RESULTS_PATH}/d-$execname"
		    file="$dir/u1-error.dat"
		    if [ -f $file ]
		    then
			line=`tail -n 1 $file`
			echo "$etai $line" >> $filecomp
		    fi
		done
		cflk=$[cflk + 1]
		lsk=$[lsk + 1]
	    done
	    anglek=$[anglek + 1]
	done
    done
fi

#########################################################

# This looks more like a makefile, echoing every executed command,
# maybe this file should be some kind of makefile.

cmd="./plots/error.py \"${RESULTS_PATH}\" \"${DEPLOY_PATH}\""
echo $cmd
eval $cmd
cmd="./plots/error_special_1.py \"${RESULTS_PATH}\" \"${DEPLOY_PATH}\""
echo $cmd
eval $cmd
cmd="./plots/error_special_2.py \"${RESULTS_PATH}\" \"${DEPLOY_PATH}\""
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

cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/std_cfl.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/transverse100_1.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/transverse100_2.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/transverse100_3.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/error_angles.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/error_special_1.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/error_etas_CPU.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/error_etas_GPU.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/error_etas_CPU_special_2.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/typical.gp"
echo $cmd
eval $cmd
cmd="gnuplot -e \"$GNUPLOT_VARS\" ./plots/error_cfls.gp"
echo $cmd
eval $cmd

export_path="${DEPLOY_PATH}/plots_export_1"
if [ ! -d "$export_path" ]
then
    mkdir $export_path

    cp -a "${DEPLOY_PATH}/plots_std_cfl/ts_cfl_std-p2.png" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_error_etas_CPU_special_2/ts_error-etas-p2-L1-sa-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_typical/ts_typical-sa-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/table_error_4/conv_rates_stats_angles-p2.dat" "${export_path}"
fi

export_path="${DEPLOY_PATH}/plots_export_2"
if [ ! -d "$export_path" ]
then
    mkdir $export_path
    cp -a "${DEPLOY_PATH}/plots_error_cfls/ts_error-cfls-p2-L1.png" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_error_cfls/ts_error-cfls-p2-Li.png" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_error_angles/ts_error-angles-p2-L1.png" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_error_angles/ts_error-angles-p2-Li.png" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_std_cfl/ts_cfl_std-p2.png" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_error_etas_CPU_special_2/ts_error-etas-p2-L1-sa-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/plots_typical/ts_typical-sa-crop.pdf" "${export_path}"
    cp -a "${DEPLOY_PATH}/table_error_4/conv_rates_stats_angles-p2.dat" "${export_path}"
fi
