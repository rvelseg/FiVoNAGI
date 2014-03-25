#!/bin/bash
#

# For comparison with CLAWPACK see: clawpack-4.6.1/apps/acoustics/2d/example1/setplot.py

ps=(1  2)
etas=(20 41 82)
angles=(    "0"     "PI/32"     "PI/16"     "PI/8"     "PI/4") 
angles_str=("00000" "PIo32"     "PIo16"     "PIo08"    "PIo04")
angles_tex=("0"     "\\\\pi/32" "\\\\pi/16" "\\\\pi/8" "\\\\pi/4")
cfls=(60 70 80 90 99)
cfls_tex=("0.60" "0.70" "0.80" "0.90" "0.99")

linestyles=(1 2 3 4 5 7 8 9 10 11 12 13 14 16 17 18 19 20 21 22)
linecolors=("dark-grey" "red" "web-green" "web-blue" "dark-magenta" "dark-cyan" "dark-orange" "dark-yellow" "royalblue" "goldenrod" "dark-spring-green" "purple" "steelblue" "dark-red" "dark-chartreuse" "orchid" "aquamarine" "brown" "yellow" "turquoise" "grey0" "grey10" "grey20" "grey30" "grey40" "grey50" "grey60" "grey70" "grey" "grey80" "grey90" "grey100" "light-red" "light-green" "light-blue" "light-magenta" "light-cyan" "light-goldenrod" "light-pink" "light-turquoise" "gold" "green" "dark-green" "spring-green" "forest-green" "sea-green" "blue" "dark-blue" "midnight-blue" "navy" "medium-blue" "skyblue" "cyan" "magenta" "dark-turquoise" "dark-pink" "coral" "light-coral" "orange-red" "salmon" "dark-salmon" "khaki" "dark-khaki" "dark-goldenrod" "beige" "olive" "orange" "violet" "dark-violet" "plum" "dark-plum" "dark-olivegreen" "orangered4" "brown4" "sienna4" "orchid4" "mediumpurple3" "slateblue1" "yellow4" "sienna1" "tan1" "sandybrown" "light-salmon" "pink" "khaki1" "lemonchiffon" "bisque" "honeydew" "slategrey" "seagreen" "antiquewhite" "chartreuse" "greenyellow" "gray" "light-gray" "light-grey" "dark-gray" "slategray" "gray0" "gray10" "gray20" "gray30" "gray40" "gray50" "gray60" "gray70" "gray80" "gray90" "gray100" "black")

RESULTS_PATH=$1
DEPLOY_PATH=$2

cd $DEPLOY_PATH

export_path="./plots_export"
if [ ! -d "$export_path" ]
then
    mkdir $export_path
fi

#######################################################

deploy_path="./plots_se_cfl"
if [ ! -d "$deploy_path" ]
then
    mkdir $deploy_path
fi

for pi in "${ps[@]}"
do
    plotstring=" "
    for etai in "${etas[@]}"
    do
	for cfli in "${cfls[@]}"
	do
	    compname="cfl_std-eta${eati}-cfl${cfli}-p${pi}"
	    filecomp="${deploy_path}/${compname}.dat"
	    if [ -f "$filecomp" ]
	    then
		rm "$filecomp"
	    fi
	    touch "$filecomp"
	    echo "# angle N T cfl std" >> $filecomp
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
		    line=`cat $file | grep "^100 "`
		    echo "$anglen $line" >> $filecomp
		fi
		anglek=$[anglek + 1]
	    done
	    plotstring="$plotstring \"$filecomp\" using 1:5 title \"$compname\" ,"
	done
    done
    if [ "$plotstring" != " " ]
    then
	plotstring="${plotstring%?}"
	gnuplot <<- gpcommands

set term png size 800,600
set xrange [-0.1:0.89]
set xtics ("0" 0,'pi/8' 0.393, 'pi/4' 0.785)
set xlabel 'theta_T'
set ylabel 'cfl std'
set output "$deploy_path/cfl_std-${pi}.png"
set title "cfl_std-${pi}.png"
plot $plotstring

gpcommands
    fi
    mv "${deploy_path}/cfl_std-${pi}.png" $export_path
done

###########################################################

deploy_path="./plots_transverse100"
if [ ! -d "$deploy_path" ]
then
    mkdir $deploy_path
fi

for pi in "${ps[@]}"
do
    plotstring=" "
    compname="transverse100-p${pi}"
    anglek=0
    for anglei in "${angles[@]}" 
    do
	for etai in "${etas[@]}"
	do
	    for cfli in "${cfls[@]}"
	    do
		execname="eta${etai}-cfl${cfli}-p${pi}-angle${angles_str[anglek]}"
		dir="${RESULTS_PATH}/d-$execname"
		file="$dir/u1-cut-0100.dat"
		if [ -f $file ]
		then
		    plotstring="$plotstring \"$file\" every ::3 using 2:3 with lines title \"$execname\" ,"
		fi
	    done
	done
	anglek=$[anglek + 1]
    done

    if [ "$plotstring" != " " ]
    then
	plotstring="${plotstring%?}"
	gnuplot <<- gpcommands

set xrange [-5:5]
#set xtics format " " nomirror
#set ytics format " " nomirror
set term png size 800,600
set output "$deploy_path/$compname.png"
set title "$compname"
plot $plotstring

gpcommands
    fi
done


for pi in "${ps[@]}"
do
    anglek=0
    for anglei in "${angles[@]}" 
    do
	lsk=0
	plotstring=" "
	plotstring2=" "
	refflag=0
	compname="transverse100-p${pi}-angle${angles_str[anglek]}"
	for etai in "${etas[@]}"
	do
	    cflk=0
	    for cfli in "${cfls[@]}"
	    do
		execname="eta${etai}-cfl${cfli}-p${pi}-angle${angles_str[anglek]}"
		len="\$\\\\eta=${etai}\$"
		len="$len, \$\\\\nu_W=${cfls_tex[cflk]}\$"
		dir="${RESULTS_PATH}/d-$execname"
		file="$dir/u1-cut-0100.dat"
		file2="$dir/u1-cut-0100.dat"
		if [ -f $file ]
		then
		    if [ $refflag == 0 ]
		    then
			plotstring="$plotstring \"$file\" every ::3 using 2:4 with lines lc rgb \"black\" lw 6 title \"analytic\" , \\
"
			plotstring2="$plotstring2 \"$file2\" every ::3 using 2:4 with lines lc rgb \"black\" lw 8 title \"analytic\" , \\
"
			refflag=1
		    fi
		    plotstring="$plotstring \"$file\" every ::3 using 2:3 with lines lc rgb \"${linecolors[lsk]}\" title \"$execname\" , \\
"	    
		    plotstring2="$plotstring2 \"$file2\" every ::3 using 2:3 with lines lc rgb \"${linecolors[lsk]}\" lw 4 title \"$len\" , \\
"
		    #lt ${linestyles[lsk]}
		    lsk=$[lsk + 1]
		fi
		cflk=$[cflk + 1]
	    done
	done
	anglek=$[anglek + 1]
	if [ "$plotstring" != " " ]
	then
	    plotstring="${plotstring:0:-5}"
	    plotstring2="${plotstring2:0:-5}"
	    cat <<- gpcommands > "$deploy_path/$compname.png.gp"
set xrange [-5:5]
#set xtics format " " nomirror
#set ytics format " " nomirror
set term png size 800,600
set output "$deploy_path/$compname.png"
plot $plotstring

gpcommands
	    gnuplot "$deploy_path/$compname.png.gp"

	    cd $deploy_path
	    cat <<- gpcommands > $compname.gp

set key outside right
set xrange [-5:5]
set yrange [-1.25e-4:1.25e-4]
set terminal postscript eps size 3.5,2.62 enhanced color \
    font 'Helvetica,10' linewidth 1
set output "$compname.eps"
plot $plotstring2
set terminal epslatex size 3.5,2.62 color colortext
set xlabel '\$\xi\$'
set ylabel '\$q_1-1\$'
set output "$compname.tex"
plot $plotstring2
set terminal epslatex size 4.5in,3.3in standalone color colortext
set output "$compname-sa.tex"
plot $plotstring2

gpcommands
	    gnuplot $compname.gp
	    cd ..
	fi
    done
done
cd $deploy_path
pdflatex transverse100-p2-anglePIo08-sa.tex
cd ..

mv $deploy_path/transverse100-p2-anglePIo08-sa.pdf $export_path

######################################################

deploy_path="./plots_error"
if [ ! -d "$deploy_path" ]
then
    mkdir $deploy_path
fi

for pi in "${ps[@]}"
do
    lsk=0
    plotstring=" "
    plotstring2=" "
    for etai in "${etas[@]}"
    do
	cflk=0
	for cfli in "${cfls[@]}"
	do
	    compname="error-eta${etai}-cfl${cfli}-p${pi}"
	    lcn="\$\\\\eta=${etai}\$"
 	    lcn="$lcn, \$\\\\nu_W=${cfls_tex[cflk]}\$"
	    filecomp="${deploy_path}/${compname}.dat"
	    if [ -f "$filecomp" ]
	    then
		rm "$filecomp"
	    fi
	    touch "$filecomp"
	    echo "# angle N T error" >> $filecomp
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
		    errline=`cat $file | grep "^100 "`
		    echo "$anglen $errline" >> $filecomp
		fi
		anglek=$[anglek + 1]
	    done
	    plotstring="$plotstring \"$filecomp\" using 1:4 title \"$compname\" ,"
	    plotstring2="$plotstring2 \"../$filecomp\" using 1:4 lc rgb \"${linecolors[lsk]}\" title \"$lcn\" ,"
	    cflk=$[cflk + 1]
	    lsk=$[lsk + 1]
	done
    done

    if [ "$plotstring" != " " ]
    then
	plotstring="${plotstring%?}"
	plotstring2="${plotstring2%?}"
	gnuplot <<- gpcommands

set logscale y
set xrange [0:0.8]
#set yrange [0:0.05]
#set xtics format " " nomirror
#set ytics format " " nomirror
set term png size 800,600
set output "$deploy_path/error-${pi}.png"
set title "error-${pi}"
plot $plotstring

gpcommands

	cd $deploy_path
	gnuplot <<- gpcommands

set logscale y
set key outside right
set xrange [-0.1:0.89]
#set yrange [0.015:0.035]
set terminal postscript eps size 3.5,2.62 enhanced color \
    font 'Helvetica,10' linewidth 1
set output "error-${pi}.eps"
plot $plotstring2
set terminal epslatex size 3.5,2.62 color colortext
set xtics ("0" 0,'\$\pi/8\$' 0.393, '\$\pi/4\$' 0.785)
set xlabel '\$\theta_T\$'
set ylabel '\$E\$'
set output "error-${pi}.tex"
plot $plotstring2
set terminal epslatex size 4.5in,3.3in standalone color colortext
set output "error-${pi}-sa.tex"
plot $plotstring2

gpcommands
	cd ..
    fi
done

cd $deploy_path
pdflatex error-1-sa.tex
pdflatex error-2-sa.tex
cd ..
mv $deploy_path/error-1-sa.pdf $export_path
mv $deploy_path/error-2-sa.pdf $export_path

#########################################################

deploy_path="./plots_typical"
if [ ! -d "$deploy_path" ]
then
    mkdir $deploy_path
fi

file="${RESULTS_PATH}/d-eta82-cfl99-p1-anglePIo08/u1-0100.dat"
plotstring=" \"$file\" every 1 nonuniform matrix using 2:1:3 notitle "
xL=`python -c "from numpy import * ; print 0.5+(100.0-(12.5/2.0)/cos(pi/8.0))*cos(pi/8.0)"`
xU=`python -c "from numpy import * ; print 12.5+(100.0-(12.5/2.0)/cos(pi/8.0))*cos(pi/8.0)"`
yL=`python -c "from numpy import * ; print 0.2-12.5/2.0+(100.0-(12.5/2.0)/cos(pi/8.0))*sin(pi/8.0)"`
yU=`python -c "from numpy import * ; print 12.5/2.0+(100.0-(12.5/2.0)/cos(pi/8.0))*sin(pi/8.0)"`
 
cd $deploy_path

gnuplot <<- gpcommands

set size ratio -1
set xrange [$xL:$xU]
set yrange [$yL:$yU]
set xlabel "x_1"
set ylabel "x_2"
set cblabel "q_1 -1"
#set cbrange [-1.5:1.5]
set term png 
set pm3d map
#set palette gray
set samples 100
set isosamples 100
set output "1.png"
splot $plotstring

set terminal postscript eps size 3.5,2.62 enhanced color \
    font 'Helvetica,10' linewidth 1
set output "1.eps"
replot

set terminal epslatex size 4.5in,3.3in standalone color colortext
set xlabel '\$x_1\$'
set ylabel '\$x_2\$'
set cblabel '\$q_1 -1\$'
set output "1-sa.tex"
replot

set terminal epslatex size 3.5,2.62 color colortext
set output "1.tex"
replot

# set term png 
# set xlabel "x_1"
# set ylabel "x_2"
# set border 4095 front linetype -1 linewidth 1.000
# set style line 100  linetype 5 linecolor rgb "#f0e442"  linewidth 0.500 pointtype 5 pointsize default pointinterval 0
# set view 40, 42, 1, 1
# set samples 100, 100
# set isosamples 100, 100
# unset surface
# set ticslevel 0
# set pm3d implicit at st
# set output "2.png"
# splot $plotstring

# set terminal postscript eps size 3.5,2.62 enhanced color \
#     font 'Helvetica,20' linewidth 2
# set output "2.eps"
# replot

# set terminal epslatex size 3.5,2.62 standalone color colortext
# set output "2-sa.tex"
# replot

# set terminal epslatex size 3.5,2.62 color colortext
# set output "2.tex"
# replot

gpcommands

pdflatex 1-sa.tex
#pdflatex 2-sa.tex

cd ..

mv $deploy_path/1-sa.pdf $export_path
