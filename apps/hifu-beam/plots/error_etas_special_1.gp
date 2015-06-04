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

    as = "10 20 30"
    etas = "20 41 82"
    cfls = "80 99"
    ps = "2"
    errors = "L1 L2 Li"
    errorsubs = '1 2 \infty'
    
    dataPath = deployPath."/dr_error_etas"
    exportPath = deployPath."/plots_error_etas_special_1"

    if (! dir_exists(exportPath) ) {
	
	system("mkdir ".exportPath)

	do for [i_err=1:words(errors)] {
	    
	    do for [i_p=1:words(ps)] {
		compnames = ""
		titles = ""
		do for [i_a=1:words(as)] {
		    do for [i_cfl=1:words(cfls)] {
			compname = sprintf("error-cfl%s-p%s-a%s", \
	    word(cfls,i_cfl), word(ps,i_p), word(as,i_a)) 
			compnames = compnames." ".compname
			titles = titles." ".word(as,i_a).",~0.".word(cfls,i_cfl)
		    }
		}

		outname = "hb_error-etas-p".word(ps,i_p)."-".word(errors,i_err)."-sa"
		unset title
		set ylabel '$E_'.word(errorsubs,i_err).'$'
		set xlabel '$\eta$'
		set yrange [0.005:0.8]
		set ytics add ('0.8' 0.8, '0.02' 0.02)
		set logscale y
		set logscale x
		set key inside right
		set xrange [18:90]
		set output exportPath."/".outname.".tex"
		set xtics ('20' 20, '40' 40, '80' 80)
		
		plot \
       for [i=1:words(compnames)] dataPath."/".word(compnames,i).".dat" \
       using 1:(column(1+i_err)) lc rgb word(linecolors,i) lw 3 title word(titles,i) w lp, \
       2*ref(x,1) lc rgb "black" lw 4 lt 1 notitle w l, \
       100*ref(x,2) lc rgb "black" lw 4 lt 2 notitle w l

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
