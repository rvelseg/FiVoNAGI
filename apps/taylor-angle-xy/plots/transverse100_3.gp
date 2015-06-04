reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    linecolors="dark-grey red web-green web-blue dark-magenta dark-cyan dark-orange dark-yellow royalblue goldenrod dark-spring-green purple steelblue dark-red dark-chartreuse orchid aquamarine brown yellow turquoise grey0 grey10 grey20 grey30 grey40 grey50 grey60 grey70 grey grey80 grey90 grey100 light-red light-green light-blue light-magenta light-cyan light-goldenrod light-pink light-turquoise gold green dark-green spring-green forest-green sea-green blue dark-blue midnight-blue navy medium-blue skyblue cyan magenta dark-turquoise dark-pink coral light-coral orange-red salmon dark-salmon khaki dark-khaki dark-goldenrod beige olive orange violet dark-violet plum dark-plum dark-olivegreen orangered4 brown4 sienna4 orchid4 mediumpurple3 slateblue1 yellow4 sienna1 tan1 sandybrown light-salmon pink khaki1 lemonchiffon bisque honeydew slategrey seagreen antiquewhite chartreuse greenyellow gray light-gray light-grey dark-gray slategray gray0 gray10 gray20 gray30 gray40 gray50 gray60 gray70 gray80 gray90 gray100 black"

    angles="00000 PIo32 PIo16 PIo08 PIo04"
    etas = "5 10 20 41 82"
    cfls = "60 70 80 90 99"
    ps = "1 2"

    amplfile = resultsPath."/ampl.dat" 
    if ( file_exists(amplfile) ) {
	ampl = system('sed -n ''s/.*AMPL*= *\([^ ]*.*\)/\1/p'' < '.amplfile) + 0
    } else {
	print "Amplitude file not found, aborting!"
	quit
    }

    exportPath = deployPath."/plots_transverse100_3"

    if (! dir_exists(exportPath) ) { 
	
	system("mkdir ".exportPath) 	
	do for [i_p=1:words(ps)] {
	    do for [i_angle=1:words(angles)] {
		files = ""
		titles = ""
		compname = "transverse100-p".word(ps,i_p)."-angle".word(angles,i_angle)
		do for [i_eta=1:words(etas)] {
		    do for [i_cfl=1:words(cfls)] {
			execname = sprintf("eta%s-cfl%s-p%s-angle%s", \
	    word(etas,i_eta), word(cfls,i_cfl), word(ps,i_p), word(angles,i_angle))
			dirname = resultsPath."/d-".execname
			frame = 98
			T_int = 0
			found = 0
			aborted = 0
			while ( found == 0 && aborted == 0 ) { 
			    frame = frame + 1
			    file = dirname."/u1-cut-".sprintf("%04i",frame).".dat"
			    if ( file_exists(file) ) {
				T = system('awk -F ''='' ''{if($1~/^\# T/){print $2}}'' '.file) + 0
				T_int = floor(T)
			    } else { aborted = 1 }
			    if ( T_int == 100 ) { 
				found = 1 
				files = files." ".file
				titles = titles." ".word(etas,i_eta).",~0.".word(cfls,i_cfl)
			    }
			    if ( frame == 120 ) { aborted = 1 }
			}
		    }
		}

		unset title
		set terminal epslatex size 4.5in,3.3in standalone color colortext
		set output exportPath."/ts_".compname."-sa.tex"
		set xrange [-5:5]
		set yrange [-ampl*1.5:ampl*1.5]
		set xlabel '$\xi$'
		set ylabel '$p''/p_0$'
		set key outside right
		plot \
       word(files,1) every ::3 using 2:4 with lines lc rgb "black" lw 8 title "analytic", \
       for [i=1:words(files)] word(files,i) every ::3 using 2:3 with lines lc rgb word(linecolors,i) lw 4 title word(titles,i)
		unset output

		system("cd ".exportPath." && latexmk -pdf -pdflatex=\"pdflatex -shell-escape -interaction=nonstopmode %O %S\" ".exportPath."/ts_".compname."-sa.tex")
		system("cd ".exportPath." && latexmk -c ".exportPath."/ts_".compname."-sa.tex")
		system("cd ".exportPath." && pdfcrop ".exportPath."/ts_".compname."-sa.pdf")  
	    }
	}
    }	
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
