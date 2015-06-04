reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    set term png size 800,600

    exportPath = deployPath."/plots_spectrum_etas"

    if (! dir_exists(exportPath) ) { system("mkdir ".exportPath) }

    as = "30 20 10"
    etas = "20 41 82"
    cfls = "60 70 80 90 99"
    cfls_tex = "0.60 0.70 0.80 0.90 0.99"
    ps = "1 2"
    mics = "2 3 4"

    do for [i_a=1:words(as)] {
        do for [i_cfl=1:words(cfls)] {
            do for [i_p=1:words(ps)] {
                do for [i_mic=1:words(mics)] {
		    micPath = exportPath."/mic".word(mics,i_mic)
		    if (! dir_exists(micPath) ) { system("mkdir ".micPath) }

		    ref = sprintf("eta82-cfl99-p2-a%s", word(as,i_a))
		    reffile = resultsPath."/d-".ref."/microphone_".word(mics,i_mic)."_spec.dat"
		    refamplfile = resultsPath."/d-".ref."/".ref.".dat" 
		    refampl = system('sed -n ''s/.*AMPL*= *\([^ ]*.*\)/\1/p'' < '.refamplfile)

		    plot_title = sprintf("etas-cfl%s-p%s-a%s", \
		 word(cfls,i_cfl), word(ps,i_p), word(as,i_a))
		    datfiles = ""
		    ampls = ""
		    execnames = ""
                    do for [i_eta=1:words(etas)] {
			execname = sprintf("eta%s-cfl%s-p%s-a%s", \
	    word(etas,i_eta), word(cfls,i_cfl), word(ps,i_p), word(as,i_a))
			dirname = "d-".execname
			datfile = resultsPath."/".dirname."/microphone_".word(mics,i_mic)."_spec.dat"
			amplfile = resultsPath."/".dirname."/".execname.".dat" 
			if ( file_exists(datfile) && file_exists(amplfile) ) {
			    ampl = system('sed -n ''s/.*AMPL*= *\([^ ]*.*\)/\1/p'' < '.amplfile)
			    execnames = execnames." ".execname
			    datfiles = datfiles." ".datfile   # WARNING: space as first character
			    ampls = ampls." ".ampl
			} else { print "WARNING: ".execname." : data file not found." }
		    }
		    set xrange [0:5.5]
		    set yrange [0.01:1]
		    set logscale y
		    set title plot_title
		    outfile = micPath."/".plot_title."-mic".word(mics,i_mic).".png"
		    set output outfile
		    plot \
	   reffile using 1:($2/refampl) with lines lw 4 lc rgb "black" title "reference", \
	   for [i=1:words(datfiles)] word(datfiles,i) using 1:($2/word(ampls,i)) \
	   with lines title word(execnames,i)
		}
	    }
	}
    }
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
