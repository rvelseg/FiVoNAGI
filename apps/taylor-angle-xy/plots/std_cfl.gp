reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    set term png size 800,600

    angles_str="00000 PIo32 PIo16 PIo08 PIo04"
    etas = "5 10 20 41 82"
    cfls = "60 70 80 90 99"
    ps = "1 2"

    exportPath = deployPath."/plots_std_cfl"

    if (! dir_exists(exportPath) ) {
	
	system("mkdir ".exportPath)
	
	do for [i_p=1:words(ps)] {
	    filecomps = ""
	    compnames = ""
	    do for [i_eta=1:words(etas)] {
		do for [i_cfl=1:words(cfls)] {
		    compname = sprintf("cfl_std-eta%s-cfl%s-p%s", \
	       word(etas,i_eta), word(cfls,i_cfl), word(ps,i_p))
		    filecomp = deployPath."/dr_std_cfl/".compname.".dat"
		    if ( file_exists(filecomp) ) {
			filecomps = filecomps." ".filecomp
			compnames = compnames." ".compname
		    }
		}
	    }
	    #set xrange [-0.1:0.89]
	    set key out right
	    set xlabel 'Theta_T'
	    set xtics ("0" 0,'pi/8' 0.393, 'pi/4' 0.785)
	    set ylabel 'cfl std'
	    set output exportPath."/ts_cfl_std-p".word(ps,i_p).".png"
	    set title "cfl_std-p".word(ps,i_p)
	    plot for [i_fc=1:words(filecomps)] word(filecomps,i_fc) using 1:6 \
	  title word(compnames,i_fc)
	}
    }	
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
