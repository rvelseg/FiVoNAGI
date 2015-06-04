reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0
ref(x,p) = x**(-p)

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    set terminal epslatex size 4.5in,3.3in standalone color colortext

    linecolors="dark-grey red web-green web-blue dark-magenta dark-orange royalblue goldenrod dark-spring-green purple steelblue dark-red dark-chartreuse dark-grey red web-green web-blue dark-magenta dark-orange royalblue goldenrod dark-spring-green purple steelblue dark-red dark-chartreuse"

    angles="00000 PIo32 PIo16 PIo08 PIo04"
    angles_tex='$0$ $\pi/32$ $\pi/16$ $\pi/8$ $\pi/4$'
    etas = "5 10 20 41 82"
    cfls = "60 70 80 90 99"
    ps = "1 2"
    errors = "L1 L2 Li"

    dataPath = deployPath."/dr_error_etas_CPU_special_1"
    exportPath = deployPath."/plots_error_etas_CPU_1"

    if (! dir_exists(exportPath) ) {

	system("mkdir ".exportPath) 

	do for [i_err=1:words(errors)] {

	do for [i_p=1:words(ps)] {
	    titles = ""
	    compnames = ""
	    do for [i_angle=1:words(angles)] {
		do for [i_cfl=1:words(cfls)] {
		    compname = sprintf("error-cfl%s-p%s-angle%s", \
	       word(cfls,i_cfl), word(ps,i_p), word(angles,i_angle)) 
		    compnames = compnames." ".compname
		    titles = titles." ".word(angles_tex,i_angle).",~0.".word(cfls,i_cfl)
		}
	    }

	    outname = "ts_error-etas-p".word(ps,i_p)."-".word(errors,i_err)."-sa"
	    set logscale y
	    set logscale x
	    set key inside right
	    set xrange [4:90]
	    set yrange [0.001:0.1]
	    set xtics add ('5' 5, '10' 10, '20' 20, '40' 40, '80' 80)
	    set ytics add ('0.003' 0.003, '0.01' 0.01, '0.03' 0.03)
	    set xlabel '$\eta$'
	    set ylabel '$E$'
	    set output exportPath."/".outname.".tex"

	    plot \
	  for [i=1:words(compnames)] dataPath."/".word(compnames,i).".dat" \
	  using 1:(column(3+i_err)) lc rgb word(linecolors,i) lw 3 title word(titles,i) w lp, \
	  0.5*ref(x,1) lw 5 title '$\eta^{-1}$' w l, \
	  ref(x,2) lw 5 title '$\eta^{-2}$' w l

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
