reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0
ref(x,p) = x**(-p)

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    set term png size 800,600

    as = "10 20 30"
    etas = "20 41 82"
    cfls = "60 70 80 90 99"
    ps = "1 2"
    errors = "L1 L2 Li"

    dataPath = deployPath."/dr_error_etas"
    exportPath = deployPath."/plots_error_etas"

    if (! dir_exists(exportPath) ) {
	
	system("mkdir ".exportPath)

	do for [i_err=1:words(errors)] {
	    
	    do for [i_p=1:words(ps)] {
		compnames = ""
		do for [i_a=1:words(as)] {
		    do for [i_cfl=1:words(cfls)] {
			compname = sprintf("error-cfl%s-p%s-a%s", \
	    word(cfls,i_cfl), word(ps,i_p), word(as,i_a)) 
			compnames = compnames." ".compname
		    }
		}

		outname = "hb_error-p".word(ps,i_p)."-".word(errors,i_err)
		set logscale y
		set logscale x
		set key out right
		set xrange [18:90]
		set output exportPath."/".outname.".png"
		set xtics add ('20' 20, '40' 40, '80' 80)
		set title outname
		set xlabel "eta"		
	    
		plot \
       for [i=1:words(compnames)] dataPath."/".word(compnames,i).".dat" \
       using 1:(column(1+i_err)) title word(compnames,i) w lp, \
       10*ref(x,1) lw 2 title '$\eta^{-1}$' w l, \
       10*ref(x,2) lw 2 title '$\eta^{-2}$' w l
	    }
	}
    }
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
