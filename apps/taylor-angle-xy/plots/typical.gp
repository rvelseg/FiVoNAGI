reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    amplfile = resultsPath."/ampl.dat" 
    if ( file_exists(amplfile) ) {
	ampl = system('sed -n ''s/.*AMPL*= *\([^ ]*.*\)/\1/p'' < '.amplfile) + 0
    } else {
	print "Amplitude file not found, aborting!"
	quit
    }
    
    exportPath = deployPath."/plots_typical"

    if (! dir_exists(exportPath) ) { 

	system("mkdir ".exportPath)

	file = resultsPath."/d-eta82-cfl99-p2-anglePIo08/u1-0105.dat"

	plotstring = " \"$file\" every 1 nonuniform matrix using 2:1:3 notitle "
	xL = system('python -c "from numpy import * ; print 0.5+(100.0-(12.5/2.0)/cos(pi/8.0))*cos(pi/8.0)"')
	xU = system('python -c "from numpy import * ; print 12.5+(100.0-(12.5/2.0)/cos(pi/8.0))*cos(pi/8.0)"')
	yL = system('python -c "from numpy import * ; print 0.2-12.5/2.0+(100.0-(12.5/2.0)/cos(pi/8.0))*sin(pi/8.0)"')
	yU = system('python -c "from numpy import * ; print 12.5/2.0+(100.0-(12.5/2.0)/cos(pi/8.0))*sin(pi/8.0)"')
 
	set size ratio -1
	set xrange [xL:xU]
	set yrange [yL:yU]
	set xlabel "x_1"
	set ylabel "x_2"
	set cblabel "p'/p_0"
	set cbrange [-1.5*ampl:1.5*ampl]
	set term png 
	set pm3d map
	#set palette gray
	set samples 100
	set isosamples 100
	set output exportPath."/ts_typical.png"
	splot file every 1 nonuniform matrix using 2:1:3 notitle 

	set terminal epslatex size 4.5in,3.3in standalone color colortext
	set xlabel '$x_1$'
	set ylabel '$x_2$'
	set cblabel '$p''/p_0$'
	set output exportPath."/ts_typical-sa.tex"
	replot

	unset output

	fixline1 = '\\usepackage{epstopdf}'
	fixline2 = '\\epstopdfDeclareGraphicsRule{.eps}{jpg}{.jpg}{convert -density 300 #1 \\OutputFile}'

	system("cd ".exportPath.' && awk ''/\\begin{document}/{print "'.fixline1.'\n'.fixline2.'"}1'' ts_typical-sa.tex > tmp.tex && mv tmp.tex ts_typical-sa.tex')

	system("cd ".exportPath.' && awk ''{ if($0~/\\put\(0,0\)/){print gensub(/\\includegraphics{([^}]+)}}/, "\\\\makebox[\\\\textwidth]{\\\\includegraphics[width=\\\\paperwidth]{\\1}}}", "g")} else{ print $0}}'' ts_typical-sa.tex > tmp.tex && mv tmp.tex ts_typical-sa.tex')

	system("cd ".exportPath." && latexmk -pdf -pdflatex=\"pdflatex -shell-escape -interaction=nonstopmode %O %S\" ts_typical-sa.tex")
	system("cd ".exportPath." && latexmk -c ts_typical-sa.tex")
	system("cd ".exportPath." && pdfcrop --margins \"-58 -36 -13 -17\" ts_typical-sa.pdf")
    }	
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
