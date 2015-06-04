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
    errors = "L1 L2 Li"

    dataPath = deployPath."/dr_error_cfls"
    exportPath = deployPath."/plots_error_cfls"

    if (! dir_exists(exportPath) ) {
	
	system("mkdir ".exportPath)

	do for [i_err=1:words(errors)] {
	    
	    do for [i_p=1:words(ps)] {
		compnames = ""
		do for [i_eta=1:words(etas)] {
		    do for [i_angle=1:words(angles)] {
			compname = sprintf("error-eta%s-p%s-angle%s", \
	    word(etas,i_eta), word(ps,i_p), word(angles,i_angle)) 
			compnames = compnames." ".compname
		    }
		}

		outname = "ts_error-cfls-p".word(ps,i_p)."-".word(errors,i_err)
		set logscale y
		set key out right
		set xrange [0.5:1.1]
		set output exportPath."/".outname.".png"
		set title outname
	    
        	plot \
              for [i=1:words(compnames)] dataPath."/".word(compnames,i).".dat" \
              using ($1/100):(column(3+i_err)) title word(compnames,i) w lp
	    }
	}
    }
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
