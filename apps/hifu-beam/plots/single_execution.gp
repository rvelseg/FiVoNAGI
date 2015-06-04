reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    set term png size 800,600

    as = "30 20 10"
    etas = "20 41 82"
    cfls = "60 70 80 90 99"
    ps = "1 2"

    reference_path = appPath."/reference_data"
    exportPath = deployPath."/plots_se"

    if (! dir_exists(exportPath) ) {

	system("mkdir ".exportPath)

	do for [i_a=1:words(as)] {
	    do for [i_eta=1:words(etas)] {
		do for [i_cfl=1:words(cfls)] {
		    do for [i_p=1:words(ps)] {
			
			execname = sprintf("eta%s-cfl%s-p%s-a%s", \
	    word(etas,i_eta), word(cfls,i_cfl), word(ps,i_p), word(as,i_a))
			dirname = "d-".execname
			ref = reference_path."/extracted/data-2-a".word(as,i_a).".csv"
			datfile = resultsPath."/".dirname."/".execname.".dat" 
			
			if ( file_exists(datfile) && file_exists(ref) ) { 
			    ampl = system('sed -n ''s/.*AMPL*= *\([^ ]*.*\)/\1/p'' < '.datfile)
			    set xrange [0:2]
			    set yrange [0:9]
			    #set xtics format " " nomirror
			    #set ytics format " " nomirror
			    set output exportPath."/hb_".execname.".png"
			    set title execname
			    plot \
	    datfile using 1:($2/ampl) with lines lc rgb "red" title execname, \
	    ref using 1:2 with lines lc rgb "black" title "Albin et al."
			}
		    }
		}
	    }
	}
    }
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
