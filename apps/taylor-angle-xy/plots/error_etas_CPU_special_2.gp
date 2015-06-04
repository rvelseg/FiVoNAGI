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

    angles="00000 PIo16 PIo04"
    angles_tex='$0$ $\pi/16$ $\pi/4$'
    etas = "5 10 20"
    cfls = "80 99"
    ps = "2"
    errors = "L1 L2 Li"
    errorsubs = '1 2 \infty'

    dataPath = deployPath."/dr_error_etas_CPU_special_2"
    exportPath = deployPath."/plots_error_etas_CPU_special_2"

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
		set xrange [4:100]
		set yrange [0.0001:0.1]
		set xtics ('5' 5, '10' 10, '20' 20, '40' 40, '80' 80)
		set ytics add ('0.003' 0.003, '0.01' 0.01, )
		set xlabel '$\eta$'
		set ylabel '$E_'.word(errorsubs,i_err).'$'
		set output exportPath."/".outname.".tex"

		plot \
       for [i=1:words(compnames)] dataPath."/".word(compnames,i).".dat" \
       using 1:(column(3+i_err)) lc rgb word(linecolors,i) lw 3 title word(titles,i) w lp, \
       0.05*ref(x,1) lc rgb "black" lw 4 lt 1 notitle w l, \
       ref(x,2) lc rgb "black" lw 4 lt 2 notitle w l

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
