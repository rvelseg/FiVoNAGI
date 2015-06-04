reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    set term png size 800,600

    exportPath = deployPath."/plots_spectrum_se"

    if (! dir_exists(exportPath) ) { system("mkdir ".exportPath) }

    as = "30 20 10"
    etas = "20 41 82"
    cfls = "60 70 80 90 99"
    ps = "1 2"
    mics = "2 3 4"

    do for [i_a=1:words(as)] {
	do for [i_eta=1:words(etas)] {
	    do fpior [i_cfl=1:words(cfls)] {
		do for [i_p=1:words(ps)] {
		    do for [i_mic=1:words(mics)] {
			micPath = exportPath."/mic".word(mics,i_mic)
			if (! dir_exists(micPath) ) { system("mkdir ".micPath) }
			execname = sprintf("eta%s-cfl%s-p%s-a%s", \
	    word(etas,i_eta), word(cfls,i_cfl), word(ps,i_p), word(as,i_a))
			dirname = "d-".execname
			datfile = resultsPath."/".dirname."/microphone_".word(mics,i_mic)."_spec.dat"
			amplfile = resultsPath."/".dirname."/".execname.".dat" 
			ampl = system('sed -n ''s/.*AMPL*= *\([^ ]*.*\)/\1/p'' < '.amplfile)
			if ( dir_exists(resultsPath."/".dirname) ) {
			    if ( file_exists(datfile) ) {
				set xrange [0:20]
				set yrange [0.001:10]
				set logscale y
				outfile = micPath."/".execname."-mic".word(mics,i_mic).".png"
				set output outfile
				plot datfile using 1:($2/ampl) with lines title execname
			    } else {
				print "WARNING: ".datfile." not found"
			    }
			} else {
			    print "WARNING: ".dirname." not found"
			}
		    }
		}
	    }
	}
    }
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
