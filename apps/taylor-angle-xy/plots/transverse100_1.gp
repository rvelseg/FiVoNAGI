reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    set term png size 800,600

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
    
    exportPath = deployPath."/plots_transverse100_1"

    if (! dir_exists(exportPath) ) { 

	system("mkdir ".exportPath)
	
	do for [i_p=1:words(ps)] {
	    files = ""
	    compname = "transverse100-p".word(ps,i_p)
	    do for [i_angle=1:words(angles)] {
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
			    }
			    if ( frame == 120 ) { aborted = 1 }
			}
		    }
		}
	    }
	    set xrange [-5:5]
	    set yrange [-ampl*1.5:ampl*1.5]
	    set output exportPath."/ts_".compname.".png"
	    set title compname

	    plot for [i=1:words(files)] word(files,i) every ::3 using 2:3 with lines notitle
	}
    }
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
