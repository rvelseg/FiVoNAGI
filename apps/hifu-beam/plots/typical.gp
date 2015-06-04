reset

dir_exists(dir) = system("[ -d '".dir."' ] && echo '1' || echo '0'") + 0
file_exists(file) = system("[ -f '".file."' ] && echo '1' || echo '0'") + 0

# these variables must be passed using -e when calling gnuplot, e.g.
#     gnuplot -e "path1='/path/to/some/place'; var='value'" thisfile.gp
# (no trailing slashes in paths)
# http://stackoverflow.com/questions/12328603
if ( exists("resultsPath") && exists("deployPath") && exists("appPath") ) {

    exportPath = deployPath."/plots_typical"

    if (! dir_exists(exportPath) ) {

	system("mkdir ".exportPath)

	frames = "0034 0054 0074 0094"
	Ts = "" 
	files = ""
	
	do for [i_f=1:words(frames)] {
	    file = resultsPath."/d-eta20-cfl99-p2-a30/u1-".word(frames,i_f).".dat"
	    T = system('awk -F ''='' ''{if($1~/^\# T/){print $2}}'' '.file) + 0
	    Tstr = sprintf("%i",T)
	    Ts = Ts." ".Tstr
	    files = files." ".file
	    
	    set term png 
	    set size ratio -1
	    set title "T = ".Tstr
	    set yrange [-0.5:0.5]
	    set xrange [(T-12.5)/50.:(T+12.5)/50.]
	    set xlabel "x_1 / F"
	    set ylabel "x_2 / F"
	    set xtics ("0" 0,'0.2' 0.2, '0.4' 0.4, '0.6' 0.6, '0.8' 0.8, '1' 1, '1.2' 1.2, '1.4' 1.4, '1.6' 1.6, '1.8' 1.8)
	    set ytics ("0" -0.5,'0.2' -0.3, '0.4' -0.1, '0.6' 0.1, '0.8' 0.3, '1' 0.5)
	    set cblabel "p'/p_0"
	    #set cbrange [-1.5:1.5]
	    set pm3d map
	    # set palette gray
	    set samples 100
	    set isosamples 100
	    set output exportPath."/hb_".word(frames,i_f).".png"
	    
	    splot file every 1 nonuniform matrix using 2:1:3 notitle
	}

	reset

	set terminal epslatex size 9in,3.3in standalone color colortext

	set size ratio -1
	set yrange [-0.5:0.5]
	set cblabel "$p'/p_0$"
	set cbrange [-0.0001:0.0001]
	#set cbrange [-1.5:1.5]
	set pm3d map
	# set palette gray

	set output exportPath."/hb_typical.tex"
	set multiplot layout 1,4 rowsfirst
	set xlabel '$x_1 / F$'
	set ylabel '$x_2 / F$'
	set xtics ("0" 0,'0.2' 0.2, '0.4' 0.4, '0.6' 0.6, '0.8' 0.8, '1' 1, '1.2' 1.2, '1.4' 1.4, '1.6' 1.6, '1.8' 1.8)
	set ytics ("0" -0.5,'0.2' -0.3, '0.4' -0.1, '0.6' 0.1, '0.8' 0.3, '1' 0.5)
	unset colorbox
	set xrange [(word(Ts,1)-12.5)/50.:(word(Ts,1)+12.5)/50.]
	set lmargin at screen 0.1
	set rmargin at screen 0.225
	set label 1 'a)' at graph 0.72,0.9 
	splot word(files,1) every 1 nonuniform matrix using 2:1:3 notitle

	unset ytics
	unset ylabel
	set xrange [(word(Ts,2)-12.5)/50.:(word(Ts,2)+12.5)/50.]
	set lmargin at screen 0.275
	set rmargin at screen 0.4
	set label 1 'b)' at graph 0.72,0.9 
	splot word(files,2) every 1 nonuniform matrix using 2:1:3 notitle

	set xrange [(word(Ts,3)-12.5)/50.:(word(Ts,3)+12.5)/50.]
	set lmargin at screen 0.45
	set rmargin at screen 0.575
	set label 1 'c)' at graph 0.72,0.9 
	splot word(files,3) every 1 nonuniform matrix using 2:1:3 notitle

	set colorbox
	set xrange [(word(Ts,4)-12.5)/50.:(word(Ts,4)+12.5)/50.]
	set lmargin at screen 0.625
	set rmargin at screen 0.75
	set label 1 'd)' at graph 0.72,0.9 
	splot word(files,4) every 1 nonuniform matrix using 2:1:3 notitle

	unset multiplot
	unset output
	set output
	set term pop

	system("cd ".exportPath." && latexmk -pdf -pdflatex=\"pdflatex -shell-escape -interaction=nonstopmode %O %S\" hb_typical.tex")
	system("cd ".exportPath." && latexmk -c hb_typical.tex")
	system("cd ".exportPath." && pdfcrop hb_typical.pdf")
	system("convert -density 300 ".exportPath."/hb_typical-inc.eps ".exportPath."/hb_typical-inc.png")
	system("diff -u image_fixes/hb_typical_fix.tex ".exportPath."/hb_typical.tex > ".exportPath."/hb_typical.diff")
	system("cd image_fixes && latexmk -pdf -pdflatex=\"pdflatex -shell-escape -interaction=nonstopmode %O %S\" hb_typical_fix.tex")
	system("cd image_fixes && latexmk -c hb_typical_fix.tex")
	system("cd image_fixes && pdfcrop --margins \"-31 -35 -100 -15\" hb_typical_fix.pdf")
	system("cd image_fixes && mv hb_typical_fix-crop.pdf ".exportPath." && rm hb_typical_fix.pdf")
    }
} else {
    print "missing variables: appPath, resultsPath or deployPath"
}
