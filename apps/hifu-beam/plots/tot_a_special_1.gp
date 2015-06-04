reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    set terminal epslatex size 4.5in,3.3in standalone color colortext

    linecolors="dark-grey red web-green web-blue dark-magenta dark-orange royalblue goldenrod dark-spring-green purple steelblue dark-red dark-chartreuse"

    as = "30 20 10"
    etas = "20 41 82"
    cfls = "80 99"
    ps = "2"

    ymina = "0.0 0.0 0.5 0.0"
    ymaxa = "9 6.5 4 2"

    xdetail_min = "0.9 0.88 0.8"
    xdetail_max = "0.98 1.08 1"

    ydetail_min = "7.2 5 2.5"
    ydetail_max = "8.2 6 2.75"

    xdetail_tmin = "0.9 0.9 0.7"
    xdetail_ts = "0.05 0.1 0.1"
    xdetail_tmax = "0.95 1 0.9"
    
    ydetail_ts = "0.4 0.5 0.1"

    
    reference_path = appPath."/reference_data"
    exportPath = deployPath."/plots_tot_a_special_1"

    if (! dir_exists(exportPath) ) {
	
	system("mkdir ".exportPath)
	
	do for [i_p=1:words(ps)] {
	    do for [i_a=1:words(as)] {
		compname = sprintf("a%s-p%s", word(as,i_a), word(ps,i_p))
		ref = reference_path."/extracted/data-2-a".word(as,i_a).".csv"
		datfiles = ""
		ampls = ""
		titles = ""
		do for [i_eta=1:words(etas)] {
		    do for [i_cfl=1:words(cfls)] {
			execname = sprintf("eta%s-cfl%s-p%s-a%s", \
	    word(etas,i_eta), word(cfls,i_cfl), word(ps,i_p), word(as,i_a))
			dirname = "d-".execname
			datfile = resultsPath."/".dirname."/".execname.".dat" 
			if ( file_exists(datfile) ) {
			    ampl = system('sed -n ''s/.*AMPL*= *\([^ ]*.*\)/\1/p'' < '.datfile)
			    datfiles = datfiles." ".datfile
			    ampls = ampls." ".ampl
			    titles = titles." ".word(etas,i_eta).",~0.".word(cfls,i_cfl)
			}
		    }
		}
		set output exportPath."/hb_".compname."-sa.tex"
		
		set size 1.0, 1.0
		set origin 0.0, 0.0
		# http://ontublog.blogspot.mx/2009/10/plots-inside-other-plots-with-gnuplot.html
		set multiplot

		# big plot options
		set size 1,1
		set origin 0,0
		set xrange [0:2]
		set xtics auto
		set yrange [word(ymina,i_a):word(ymaxa,i_a)]
		set ytics 1
		set key inside top right
		set xlabel '$x_1 / F$'
		set ylabel '$p''/(p_0A) $'
		unset title

		plot \
       ref using 1:2 with lines lw 6 lc rgb "black" title 'Albin {\em et al.}', \
       for [i_df=1:words(datfiles)] word(datfiles,i_df) using 1:($2/word(ampls,i_df)) with lines lc rgb word(linecolors,i_df) lw 3 title word(titles,i_df)
		
       	      	# small plot options
       	      	set size 0.4,0.4
       	      	set origin 0.08,0.55
       	      	set xrange [word(xdetail_min,i_a):word(xdetail_max,i_a)]
		set xtics word(xdetail_tmin,i_a),word(xdetail_ts,i_a),word(xdetail_tmax,i_a)
       	      	set yrange [word(ydetail_min,i_a):word(ydetail_max,i_a)]
		set ytics word(ydetail_ts,i_a)
       	      	set xlabel ""
       	      	set ylabel ""
		unset key

       	      	plot \
		    ref using 1:2 with lines lw 6 lc rgb "black" title 'Albin {\em et al.}', \
		    for [i_df=1:words(datfiles)] word(datfiles,i_df) using 1:($2/word(ampls,i_df)) with lines lc rgb word(linecolors,i_df) lw 3 title word(titles,i_df)
		
		unset multiplot
		
		# if this line is omitted tex file is incomplete when read by pdflatex
		unset output

		system("cd ".exportPath." && latexmk -pdf -pdflatex=\"pdflatex -shell-escape -interaction=nonstopmode %O %S\" ".exportPath."/hb_".compname."-sa.tex")
		system("cd ".exportPath." && latexmk -c ".exportPath."/hb_".compname."-sa.tex")
		system("cd ".exportPath." && pdfcrop ".exportPath."/hb_".compname."-sa.pdf")  
	    }
	}
    }
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
