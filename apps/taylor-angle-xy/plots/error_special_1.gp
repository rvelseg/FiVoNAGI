reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    set terminal epslatex size 9in,3.3in standalone color colortext

    linecolors="dark-grey red web-green web-blue dark-magenta dark-cyan dark-orange dark-yellow royalblue goldenrod dark-spring-green purple steelblue dark-red dark-chartreuse orchid aquamarine brown yellow turquoise grey0 grey10 grey20 grey30 grey40 grey50 grey60 grey70 grey grey80 grey90 grey100 light-red light-green light-blue light-magenta light-cyan light-goldenrod light-pink light-turquoise gold green dark-green spring-green forest-green sea-green blue dark-blue midnight-blue navy medium-blue skyblue cyan magenta dark-turquoise dark-pink coral light-coral orange-red salmon dark-salmon khaki dark-khaki dark-goldenrod beige olive orange violet dark-violet plum dark-plum dark-olivegreen orangered4 brown4 sienna4 orchid4 mediumpurple3 slateblue1 yellow4 sienna1 tan1 sandybrown light-salmon pink khaki1 lemonchiffon bisque honeydew slategrey seagreen antiquewhite chartreuse greenyellow gray light-gray light-grey dark-gray slategray gray0 gray10 gray20 gray30 gray40 gray50 gray60 gray70 gray80 gray90 gray100 black"

    angles="00000 PIo32 PIo16 PIo08 PIo04"
    etas = "5 10 20 41 82"
    cfls = "60 70 80 90 99"
    ps = "1 2"
    errors = "L1 L2 Li"
    
    exportPath = deployPath."/plots_error_special_1"
    dataPath = deployPath."/dr_error"

    if (! dir_exists(exportPath) ) { 
	
	system("mkdir ".exportPath)

	do for [i_err=1:words(errors)] {

	    do for [i_p=1:words(ps)] {
		titles = ""
		compnames = ""
		do for [i_eta=1:words(etas)] {
		    do for [i_cfl=1:words(cfls)] {
			compname = sprintf("error-eta%s-cfl%s-p%s", \
	    word(etas,i_eta), word(cfls,i_cfl), word(ps,i_p)) 
			compnames = compnames." ".compname
			titles = titles." ".word(etas,i_eta).",~0.".word(cfls,i_cfl)
		    }
		}

		outname = "ts_error-etas-p".word(ps,i_p)."-".word(errors,i_err)."-sa"
		set logscale y
		set key outside right
		set xrange [-0.1:0.89]
		set xtics ("0" 0,'$\pi/8$' 0.393, '$\pi/4$' 0.785)
		set xlabel '$\theta_T$'
		set ylabel '$E$'
		set output exportPath."/".outname.".tex"

		plot \
       for [i=1:words(compnames)] dataPath."/".word(compnames,i).".dat" \
       using 1:(column(3+i_err)) lc rgb word(linecolors,i) title word(titles,i) w lp
	    
		unset output
	    
		# replace `system` with `print sprintf` if you wan to see the commands to be executed
		system("cd ".exportPath." && latexmk -pdf -pdflatex=\"pdflatex -shell-escape -interaction=nonstopmode %O %S\" ".outname.".tex")
		system("cd ".exportPath." && latexmk -c ".outname.".tex")
		system("cd ".exportPath." && pdfcrop ".outname.".pdf")
	    }
	}
    }
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
