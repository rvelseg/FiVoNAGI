#!/bin/bash
#

as=(30)
pi=2
etai=20
cfli=99

RESULTS_PATH=$1
DEPLOY_PATH=$2

cd $DEPLOY_PATH

deploy_path_frames="./video_frames_special"
if [ ! -d "$deploy_path_frames" ]
then
    mkdir $deploy_path_frames
fi

deploy_path_videos="./videos_special"
if [ ! -d "$deploy_path_videos" ]
then
    mkdir $deploy_path_videos
fi

for ai in "${as[@]}"
do
    compname="v-cfls-eta${etai}-p${pi}-a${ai}-2"
    if [ -d "${deploy_path_frames}/$compname" ]
    then
	echo "$(date +"%x %T") ::warning::deploy destination already exists. Not overwritten" 
	echo "${deploy_path_frames}/$compname"
    else
	mkdir "${deploy_path_frames}/${compname}"

	plotstring=" "

	for framei in {0..115}
	do
	    echo "frame: ${framei}"
	    offset="0.0"
	    xwidth="25.0"
	    cscale="0.0005"
	    frameif=`printf "%04d" $framei`
	    execname="eta${etai}-cfl${cfli}-p${pi}-a${ai}"
	    dir="${RESULTS_PATH}/d-$execname"
	    if [ -f "$dir/u1-${frameif}.dat" ]
	    then
		TIME=$(sed -n 's/.*T*= *\([^ ]*.*\)/\1/p' < "$dir/u1-${frameif}.dat")
		echo $TIME
		plotstring=" \"$dir/u1-${frameif}.dat\" every 10 nonuniform matrix using (\$2*50.0):((\$1+0.5)*50.0):3 notitle, \
\"$dir/u1-${frameif}.dat\" every 10 nonuniform matrix using (\$2*50.0):((-\$1-0.5)*50.0):3 notitle "
	    else
		echo "datafile not found: $dir/u1-${frameif}.dat"
	    fi
	    if [ "$plotstring" != " " ]
	    then
		out="${deploy_path_frames}/${compname}/${compname}-zoom1-${frameif}"
		comp=$(python -c "print ${TIME} > 12.5")
		if [ $comp == "True" ]
		then
		    xdls=$(python -c "from numpy import * ; print (${TIME}-${offset}-${xwidth}/2.0)")
		    xdrs=$(python -c "from numpy import * ; print (${TIME}-${offset}+${xwidth}/2.0)")
		else
		    xdls=0
		    xdrs="25.0" 
		fi
		aw=$(python -c "from numpy import * ; print ${ai}*1.0")
		cat <<- gpcommands > "${out}.gp"
reset

cd "$(pwd)"

set term png font arial 24 size 1200,1200 nocrop
set output "${out}.png"
set tmargin at screen 0.95
set rmargin at screen 0.95
set lmargin at screen 0.1
set bmargin at screen 0.1
set pm3d map
set size ratio -1
unset colorbox
set cbrange [-$cscale:$cscale]

set object 1 rectangle from 0,-50 to 110,50 fillcolor rgb "#b52000" behind
# set object 2 rectangle from -15,-$aw to 0,-$aw-1 fillcolor rgb "#000000" behind
# set object 3 rectangle from -15,$aw to 0,$aw+1 fillcolor rgb "#000000" front
set arrow 1 from -8.4,0 to -8.4,$aw heads front
set arrow 2 from -5,-$aw to 0,-$aw nohead lw 10 
set arrow 3 from -5,$aw to 0,$aw nohead lw 10 
set label 1 'a' at -10.1,$aw/2 
set label 2 'Axis units are wavelegths' at 0,-62     

set xrange [0:110]
set yrange [-50:50]
splot $plotstring, '-' with lines lc rgb "black" notitle, '-' with lines lc rgb "white" notitle, 
$xdls 0 1
$xdls 50 1
$xdrs 50 1
$xdrs 0 1
$xdls 0 1
e
50 -50 1
50 0 1
e

gpcommands
gnuplot "${out}.gp"
	    fi
	done

	if [ -f "${deploy_path_frames}/${compname}/${compname}-zoom1-0000.png" ]
	then
	    echo "video compilation started"
	    rm "./$deploy_path_videos/${compname}-zoom1-2.avi"
	    mencoder mf://${deploy_path_frames}/${compname}/${compname}-zoom1-*.png -mf w=800:h=600:fps=5:type=png -ovc copy -oac copy -o ./$deploy_path_videos/${compname}-zoom1-2.avi && echo "video compilation completed"
	fi
    fi
done

