reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    set term png size 800,600

    linecolors="dark-grey red web-green web-blue dark-magenta dark-cyan dark-orange dark-yellow royalblue goldenrod dark-spring-green purple steelblue dark-red dark-chartreuse orchid aquamarine brown yellow"

    exportPath = deployPath."/plots_spectrum_special_1"

    if (! dir_exists(exportPath) ) {

	system("mkdir ".exportPath) 

	as = "30"
	etas = "20 41"
	cfls = "80 99"
	cfls_tex = "0.80 099"
	ps = "2"
	mics = "3"

	plot_title = "spectrum_special_1"

	# this is to be consistent with other plotting scripts
	i_a = 1
	i_p = 1
	i_mic = 1

	datfiles = ""
	ampls = ""
	execnames = ""
	tex_titles = ""
	
	amplfile = resultsPath."/d-eta82-cfl99-p2-a30/eta82-cfl99-p2-a30.dat" 
	if ( file_exists(amplfile) ) {
	    ampl_ref = system('sed -n ''s/.*AMPL*= *\([^ ]*.*\)/\1/p'' < '.amplfile)
	} else { print "WARNING: ".amplfile." : (ampl) data file not found." }
	
	do for [i_eta=1:words(etas)] {
	    do for [i_cfl=1:words(cfls)] {
		execname = sprintf("eta%s-cfl%s-p%s-a%s", \
	   word(etas,i_eta), word(cfls,i_cfl), word(ps,i_p), word(as,i_a))
		dirname = "d-".execname
		datfile = resultsPath."/".dirname."/microphone_".word(mics,i_mic)."_spec.dat"
		amplfile = resultsPath."/".dirname."/".execname.".dat" 
		if ( file_exists(datfile) && file_exists(amplfile) ) {
		    ampl = system('sed -n ''s/.*AMPL*= *\([^ ]*.*\)/\1/p'' < '.amplfile)
		    
		    # a blank space is the first character of the
		    # following strings, but `word` and `words` functions
		    # have no problems with that, see `help words`
		    execnames = execnames." ".execname
		    datfiles = datfiles." ".datfile
		    ampls = ampls." ".ampl
		    tex_titles = tex_titles." ".word(etas,i_eta).",~0.".word(cfls,i_cfl)

		    } else { print "WARNING: ".execname." : data file not found." }
	    }
	}

	set xrange [0:5.5]
	set yrange [0.01:1]
	set logscale y
	outfile = exportPath."/hb_".plot_title.".png"
	set output outfile
	plot \
      for [i=1:words(datfiles)] word(datfiles,i) using 1:($2/word(ampls,i)) \
      with lines title word(execnames,i)
	
	set terminal epslatex size 4.5in,3.3in standalone color colortext
	set xlabel "freq."
	set ylabel "$p'/(p_0 A)$"

	outfile = exportPath."/hb_".plot_title."_sa.tex"
	set output outfile

	set size 1.0, 1.0
	set origin 0.0, 0.0
	set multiplot

	plot \
      resultsPath."/d-eta82-cfl99-p2-a30/microphone_3_spec.dat" using 1:($2/ampl_ref) \
      with lines lc rgb "black" lw 5 title "82, 0.99", \
      for [i=1:words(datfiles)] word(datfiles,i) using 1:($2/word(ampls,i)) \
      with lines lc rgb word(linecolors,i) lw 3 title word(tex_titles,i)

	# small plot options
	set size 0.4,0.3
	set origin 0.3,0.63
	set xrange [0.9:1.1]
	set xtics ('0.9' 0.9, '1.0' 1, '1.1' 1.1)
	set yrange [0.3:0.5]
	set ytics ('0.3' 0.3, '0.4' 0.4, '0.5' 0.5)
	set xlabel ""
	set ylabel ""
	unset key
	
	plot \
      resultsPath."/d-eta82-cfl99-p2-a30/microphone_3_spec.dat" using 1:($2/ampl_ref) \
      with lines lc rgb "black" lw 6 title "82, 0.99", \
      for [i=1:words(datfiles)] word(datfiles,i) using 1:($2/word(ampls,i)) \
      with lines lc rgb word(linecolors,i) lw 3 title word(tex_titles,i)

    	unset multiplot
	# if this line is omitted tex file is incomplete when read by pdflatex
	unset output
	
	system("cd ".exportPath." && pdflatex -shell-escape -interaction=nonstopmode ".outfile)
	outfilepdf = exportPath."/hb_".plot_title."_sa.pdf"
	system("cd ".exportPath." && pdfcrop ".outfilepdf)
    }	
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
