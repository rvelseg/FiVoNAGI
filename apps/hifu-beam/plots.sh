#!/bin/bash
#

# For comparison with CLAWPACK see: clawpack-4.6.1/apps/acoustics/2d/example1/setplot.py

as=(    30    20    10    05)
ymina=("0.0" "0.0" "0.6" "0.0")
ymaxa=("9"   "6"   "2.8"   "2")
etas=(20 41 82)
cfls=(     60     70     80     90     99)
cfls_tex=("0.60" "0.70" "0.80" "0.90" "0.99")
ps=(1 2)

linecolors=("dark-grey" "red" "web-green" "web-blue" "dark-magenta" "dark-cyan" "dark-orange" "dark-yellow" "royalblue" "goldenrod" "dark-spring-green" "purple" "steelblue" "dark-red" "dark-chartreuse" "orchid" "aquamarine" "brown" "yellow" "turquoise" "grey0" "grey10" "grey20" "grey30" "grey40" "grey50" "grey60" "grey70" "grey" "grey80" "grey90" "grey100" "light-red" "light-green" "light-blue" "light-magenta" "light-cyan" "light-goldenrod" "light-pink" "light-turquoise" "gold" "green" "dark-green" "spring-green" "forest-green" "sea-green" "blue" "dark-blue" "midnight-blue" "navy" "medium-blue" "skyblue" "cyan" "magenta" "dark-turquoise" "dark-pink" "coral" "light-coral" "orange-red" "salmon" "dark-salmon" "khaki" "dark-khaki" "dark-goldenrod" "beige" "olive" "orange" "violet" "dark-violet" "plum" "dark-plum" "dark-olivegreen" "orangered4" "brown4" "sienna4" "orchid4" "mediumpurple3" "slateblue1" "yellow4" "sienna1" "tan1" "sandybrown" "light-salmon" "pink" "khaki1" "lemonchiffon" "bisque" "honeydew" "slategrey" "seagreen" "antiquewhite" "chartreuse" "greenyellow" "gray" "light-gray" "light-grey" "dark-gray" "slategray" "gray0" "gray10" "gray20" "gray30" "gray40" "gray50" "gray60" "gray70" "gray80" "gray90" "gray100" "black")

APP_PATH=`pwd`
RESULTS_PATH=$1
DEPLOY_PATH=$2

REFERENCE_PATH="${APP_PATH}/reference_data/"
echo "REFERENCE_PATH: $REFERENCE_PATH"

cd $DEPLOY_PATH

export_path="./plots_export"
if [ ! -d "$export_path" ]
then
    mkdir $export_path
fi

###################################################-

deploy_path="./plots_se"
if [ ! -d "$deploy_path" ]
then
    mkdir $deploy_path
fi

for pi in "${ps[@]}"
do
    for ai in "${as[@]}"
    do
	for etai in "${etas[@]}"
	do
	    for cfli in "${cfls[@]}"
	    do
		execname="eta${etai}-cfl${cfli}-p${pi}-a${ai}"
		dir="d-$execname"
		ref="${REFERENCE_PATH}/extracted/data-2-a$ai"
		plotstring=" "
		datfile="${RESULTS_PATH}/$dir/${execname}.dat"
		if [ -f "${datfile}" ]
		then
		    AMPL=$(sed -n 's/.*AMPL*= *\([^ ]*.*\)/\1/p' < "${datfile}")
		    plotstring="$plotstring \"${datfile}\" using 1:(\$2/${AMPL}) with lines lc rgb \"red\" title \"$execname\" ,"
		fi
		if [ "$plotstring" != " " ]
		then

		    if [ -f "$ref.csv" ]
		    then
			plotstring=" \"$ref.csv\" using 1:2 with lines lc rgb \"black\" title \"Albin et al.\" , $plotstring"
		    fi

		    plotstring="${plotstring%?}"
		    cat <<- gpcommands > "$deploy_path/$execname.png.gp"

set xrange [0:2]
set yrange [0:9]
#set xtics format " " nomirror
#set ytics format " " nomirror
set term png size 800,600
set output "$deploy_path/$execname.png"
set title "$execname"
plot $plotstring

gpcommands
		    gnuplot "$deploy_path/$execname.png.gp"
		fi
	    done
	done
    done
done

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
	    filecomp="${DEPLOY_PATH}/${deploy_path}/${compname}.dat"
	    if [ -f "$filecomp" ]
	    then
		rm "$filecomp"
	    fi
	    touch "$filecomp"
	    echo "# a N T cfl std" >> $filecomp
	    for ai in "${as[@]}" 
	    do
		execname="eta${etai}-cfl${cfli}-p${pi}-a${ai}"
		dir="d-$execname"
		file="${RESULTS_PATH}/$dir/u1-cfl.dat"
		if [ -f $file ]
		then
		    line=`cat $file | grep "^100 "`
		    echo "${ai} $line" >> $filecomp
		fi
	    done
	    plotstring="$plotstring \"$filecomp\" using 1:5 title \"$compname\" ,"
	done
    done
    if [ "$plotstring" != " " ]
    then
	plotstring="${plotstring%?}"
	cat <<- gpcommands > "$deploy_path/cfl_std-${pi}.gp"

set term png size 800,600
#set xrange [-0.1:0.89]
set xlabel 'a'
set ylabel 'cfl std'
set output "$deploy_path/cfl_std-${pi}.png"
set title "cfl_std-${pi}.png"
plot $plotstring

gpcommands

	gnuplot "$deploy_path/cfl_std-${pi}.gp"
    fi
    mv "${deploy_path}/cfl_std-${pi}.png" $export_path
done


##############################################################

deploy_path="./plots_cfls"
if [ ! -d "$deploy_path" ]
then
    mkdir $deploy_path
fi

for pi in "${ps[@]}"
do
    for ai in "${as[@]}"
    do
	for etai in "${etas[@]}"
	do
	    ref="${REFERENCE_PATH}/extracted/data-2-a$ai"
	    compname="cfls-eta${etai}-p${pi}-a${ai}"
	    plotstring=" "

	    for cfli in "${cfls[@]}"
	    do
		execname="eta${etai}-cfl${cfli}-p${pi}-a${ai}"
		dir="${RESULTS_PATH}/d-$execname"
		if [ -d "$dir" ] 
		then
		    AMPL=$(sed -n 's/.*AMPL*= *\([^ ]*.*\)/\1/p' < "${dir}/${execname}.dat")
		    plotstring="$plotstring \"$dir/$execname.dat\" using 1:(\$2/${AMPL}) with lines title \"$execname\" ,"
		fi
	    done
	    if [ "$plotstring" != " " ]
	    then

		if [ -f "$ref.csv" ]
		then
		    plotstring=" \"$ref.csv\" using 1:2 with lines lw 3 lc rgb \"black\" title \"Albin et al.\" , $plotstring"
		fi

		plotstring="${plotstring%?}"
		gnuplot <<- gpcommands

set xrange [0:2]
set yrange [0:9]
set term png size 800,600
set output "$deploy_path/$compname.png"
set title "$compname"
plot $plotstring

gpcommands
	    fi
	done
    done
done

deploy_path="./plots_etas"
if [ ! -d "$deploy_path" ]
then
    mkdir $deploy_path
fi

for pi in "${ps[@]}"
do
    for ai in "${as[@]}"
    do
	for cfli in "${cfls[@]}"
	do
	    ref="${REFERENCE_PATH}/extracted/data-2-a$ai"
	    compname="etas-cfl${cfli}-p${pi}-a${ai}"
	    plotstring=" "

	    for etai in "${etas[@]}"
	    do
		execname="eta${etai}-cfl${cfli}-p${pi}-a${ai}"
		dir="${RESULTS_PATH}/d-$execname"
		if [ -d "$dir" ]
		then
		    AMPL=$(sed -n 's/.*AMPL*= *\([^ ]*.*\)/\1/p' < "${dir}/${execname}.dat")
		    plotstring="$plotstring \"$dir/$execname.dat\" using 1:(\$2/${AMPL}) with lines title \"$execname\" ,"
		fi
	    done
	    if [ "$plotstring" != " " ]
	    then

		if [ -f "$ref.csv" ]
		then
		    plotstring=" \"$ref.csv\" using 1:2 with lines lw 3 lc rgb \"black\" title \"Albin et al.\" , $plotstring"
		fi

		plotstring="${plotstring%?}"
		gnuplot <<- gpcommands

set xrange [0:2]
set yrange [0:9]
set term png size 800,600
set output "$deploy_path/$compname.png"
set title "$compname"
plot $plotstring

gpcommands
	    fi
	done
    done
done

###############################################################

deploy_path="./plots_ps"
if [ ! -d "$deploy_path" ]
then
    mkdir $deploy_path
fi

for cfli in "${cfls[@]}"
do
    for ai in "${as[@]}"
    do
	for etai in "${etas[@]}"
	do
	    ref="${REFERENCE_PATH}/extracted/data-2-a$ai"
	    compname="ps-eta${etai}-cfl${cfli}-a${ai}"
	    plotstring=" "
	    for pi in "${ps[@]}"
	    do
		execname="eta${etai}-cfl${cfli}-p${pi}-a${ai}"
		dir="${RESULTS_PATH}/d-$execname"
		if [ -d "$dir" ] 
		then
		    AMPL=$(sed -n 's/.*AMPL*= *\([^ ]*.*\)/\1/p' < "${dir}/${execname}.dat")
		    plotstring="$plotstring \"$dir/$execname.dat\" using 1:(\$2/${AMPL}) with lines title \"$execname\" ,"
		fi
	    done
	    if [ "$plotstring" != " " ]
	    then

		if [ -f "$ref.csv" ]
		then
		    plotstring=" \"$ref.csv\" using 1:2 with lines lw 3 lc rgb \"black\" title \"Albin et al.\" , $plotstring"
		fi

		plotstring="${plotstring%?}"
		gnuplot <<- gpcommands

set xrange [0:2]
set yrange [0:9]
set term png size 800,600
set output "$deploy_path/$compname.png"
set title "$compname"
plot $plotstring

gpcommands
	    fi
	done
    done
done

######################################################

deploy_path="./plots_tot_a"
if [ ! -d "$deploy_path" ]
then
    mkdir $deploy_path
fi


for pi in "${ps[@]}"
do
    ak=0
    for ai in "${as[@]}"
    do
	compname="a${ai}-p${pi}"
	ref="${REFERENCE_PATH}/extracted/data-2-a$ai"
	plotstring=" "
	plotstring2=" "
	lsk=0
	for etai in "${etas[@]}"
	do
	    cflk=0
	    for cfli in "${cfls[@]}"
	    do
		execname="eta${etai}-cfl${cfli}-p${pi}-a${ai}"
 		len="\$\\\\nu_W=${cfls_tex[cflk]}\$"
		dir="${RESULTS_PATH}/d-$execname"
		if [ -d "$dir" ] 
		then
		    AMPL=$(sed -n 's/.*AMPL*= *\([^ ]*.*\)/\1/p' < "${dir}/${execname}.dat")
		    plotstring="$plotstring \"$dir/$execname.dat\" using 1:(\$2/${AMPL}) with lines title \"$execname\" , \\
"
		    plotstring2="$plotstring2 \"$dir/$execname.dat\" using 1:(\$2/${AMPL}) with lines lc rgb \"${linecolors[lsk]}\" lw 3 title \"$len\" , \\
"
		    lsk=$[lsk + 1]
		fi
		cflk=$[cflk + 1]
	    done
	done
	if [ "$plotstring" != " " ]
	then
	    if [ -f "$ref.dat" ]
	    then
		plotstring=" \"$ref.dat\" using 1:2 with lines lw 3 lc rgb \"black\" title \"Albin et al.\" , \\
$plotstring"
		plotstring2=" \"$ref.dat\" using 1:2 with lines lw 6 lc rgb \"black\" title \"Albin {\\\\em et al.}\" , \\
$plotstring2"
	    fi

	    plotstring="${plotstring:0:-5}"
	    plotstring2="${plotstring2:0:-5}"
	    cat <<- gpcommands > "$deploy_path/$compname.gp"

set xrange [0:2]
set yrange [${ymina[ak]}:${ymaxa[ak]}]
set term png size 800,600
set output "$deploy_path/$compname.png"
set title "$compname"
plot $plotstring

set terminal postscript eps size 3.5,2.62 enhanced color \
    font 'Helvetica,10' linewidth 1
set output "$deploy_path/$compname.eps"
plot $plotstring

gpcommands
	    gnuplot "$deploy_path/$compname.gp"

	    cd $deploy_path
	    echo $plotstring2
	    cat <<- gpcommands > "$compname.tex.gp"

#set key outside right
set xrange [0:2]
set yrange [${ymina[ak]}:${ymaxa[ak]}]
set terminal epslatex size 3.5,2.62 color colortext
set xlabel '\$x_1 / F\$'
set ylabel '\$q_1 / A \$'
set output "$compname.tex"
plot $plotstring2

set terminal epslatex size 4.5in,3.3in standalone color colortext
set output "$compname-sa.tex"
plot $plotstring2

gpcommands
	    gnuplot "$compname.tex.gp"
	    cd ..
	fi
	if [ $ai != "05" ]
	then
    	    cd $deploy_path
	    pdflatex -interaction=nonstopmode "${compname}-sa.tex"
    	    if [ $? == 0 ]
		then
    		mv "../${deploy_path}/${compname}-sa.pdf" "../$export_path"
	    fi
	    cd ..
	fi
	ak=$[ak + 1]
    done
done

#########################################################

deploy_path="./plots_typical"
if [ ! -d "$deploy_path" ]
then
    mkdir $deploy_path
fi

pstrings=( )
psk=0
frames=(20 40 60 80)
for framei in "${frames[@]}"
do
    frameif=`printf "%04d" $framei`
    file="${RESULTS_PATH}/d-eta20-cfl99-p2-a30/u1-${frameif}.dat"
    plotstring=" \"$file\" every 1 nonuniform matrix using 2:1:3 notitle "
    
    cd $deploy_path

    gnuplot <<- gpcommands

set size ratio -1
set yrange [-0.5:0.5]
set xrange [(${framei}-12.5)/50.:(${framei}+12.5)/50.]
set xlabel "x_1 / F"
set ylabel "x_2 / F"
set xtics ("0" 0,'0.2' 0.2, '0.4' 0.4, '0.6' 0.6, '0.8' 0.8, '1' 1, '1.2' 1.2, '1.4' 1.4, '1.6' 1.6, '1.8' 1.8)
set ytics ("0" -0.5,'0.2' -0.3, '0.4' -0.1, '0.6' 0.1, '0.8' 0.3, '1' 0.5)
set cblabel "q_1-1"
#set cbrange [-1.5:1.5]
set term png 
set pm3d map
# set palette gray
set samples 100
set isosamples 100
set output "${framei}.png"
splot $plotstring

gpcommands

    cd ..

    pstrings[psk]=$plotstring
    psk=$[psk + 1]
done

cd $deploy_path

gnuplot <<- gpcommands

reset

set size ratio -1
set yrange [-0.5:0.5]
set cblabel "q_1-1"
set cbrange [-0.0001:0.0001]
#set cbrange [-1.5:1.5]
set pm3d map
# set palette gray

set terminal epslatex size 9in,3.3in standalone color colortext
set output "mp-sa.tex"
set multiplot layout 1,4 rowsfirst
set xlabel '\$x_1 / F\$'
set ylabel '\$x_2 / F\$'
set xtics ("0" 0,'0.2' 0.2, '0.4' 0.4, '0.6' 0.6, '0.8' 0.8, '1' 1, '1.2' 1.2, '1.4' 1.4, '1.6' 1.6, '1.8' 1.8)
set ytics ("0" -0.5,'0.2' -0.3, '0.4' -0.1, '0.6' 0.1, '0.8' 0.3, '1' 0.5)
set cblabel '\$q_1 -1\$'
unset colorbox
set xrange [(${frames[0]}-12.5)/50.:(${frames[0]}+12.5)/50.]
set lmargin at screen 0.1
set rmargin at screen 0.225
set label 1 'a)' at graph 0.72,0.9 
splot ${pstrings[0]}

unset ytics
unset ylabel
set xrange [(${frames[1]}-12.5)/50.:(${frames[1]}+12.5)/50.]
set lmargin at screen 0.275
set rmargin at screen 0.4
set label 1 'b)' at graph 0.72,0.9 
splot ${pstrings[1]}

set xrange [(${frames[2]}-12.5)/50.:(${frames[2]}+12.5)/50.]
set lmargin at screen 0.45
set rmargin at screen 0.575
set label 1 'c)' at graph 0.72,0.9 
splot ${pstrings[2]}

set colorbox
set xrange [(${frames[3]}-12.5)/50.:(${frames[3]}+12.5)/50.]
set lmargin at screen 0.625
set rmargin at screen 0.75
set label 1 'd)' at graph 0.72,0.9 
splot ${pstrings[3]}
unset multiplot
set output
set term pop

gpcommands

latex mp-sa.tex 
dvips mp-sa.dvi

convert -density 300 mp-sa-inc.eps mp-sa-inc.png

cd ${APP_PATH}
python error.py ${RESULTS_PATH} ${DEPLOY_PATH} ${REFERENCE_PATH}

# TODO:
# remove white space of some images
# pdfcrop --margins "-55 -30 -11 -13" 1.pdf
# pdfcrop --margins "-30 -30 -90 -10" 4.pdf

