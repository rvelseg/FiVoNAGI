#!/bin/bash
#

ps=(1 2)
etas=(5 10 20 41 82)
angles=(    "0"     "PI/32" "PI/16" "PI/8"  "PI/4") 
angles_str=("00000" "PIo32" "PIo16" "PIo08" "PIo04")
cfls=(99 90 80 70 60)

ROOT=$1
RESULTS_PATH=$2
EXPORT=$3

if [ ${EXPORT} ]
then
    EXPORT_STR="-D EXPORT"
    if [ ! -d "${RESULTS_PATH}" ]
    then
    echo "RESULTS_PATH directory does not exist." 
    exit 4
    fi
else
    EXPORT_STR=""
    deploy_path="."
fi

for pi in "${ps[@]}"
do
    for etai in "${etas[@]}"
    do
	if [ "$etai" == 5 ]
	then
	    etain=1
	elif [ "$etai" == 10 ]
	then
	    etain=2
	elif [ "$etai" == 20 ]
	then
	    etain=4
	elif [ "$etai" == 41 ]
	then
	    etain=8
	elif [ "$etai" == 82 ]
	then
	    etain=16
	fi
	for cfli in "${cfls[@]}"
	do
	    anglek=0
	    for anglei in "${angles[@]}"
	    do
		anglelow="$(echo $anglei | tr '[A-Z]' '[a-z]')"
		cos=`python -c "from numpy import * ; print cos($anglelow)"`
		sin=`python -c "from numpy import * ; print sin($anglelow)"`
		#shouldn't this whole script be in python?

		if [ ${EXPORT} ]
		then
		    execname="eta${etai}-cfl${cfli}-p${pi}-angle${angles_str[anglek]}"
		    deploy_path="${RESULTS_PATH}/d-$execname"
		    if [ ! -d ${deploy_path} ]
		    then
			mkdir ${deploy_path} || exit 1
		    else
		    echo "$(date +"%x %T") ::warning::deploy destination already exists for: $execname. Not overwritten" 
		    continue
		    fi
		fi
 		echo "$(date +"%x %T") ::begin  ::$execname" 
		echo `uname -a`
		echo `nvcc --version`
		command="nvcc -D ROOT=${ROOT} -D DEPLOY=${deploy_path} ${EXPORT_STR} -D SIN=$sin -D COS=$cos -D PRECISION=$pi -D ANGLE=$anglei -D CFLW=$cfli -D ETAC=$etain -o FiVoNAGI -lcuda -lcudart  -lm -lGL -lGLU -lglut -lpthread -arch=sm_13 ./driver.cu"
		echo $command
		eval $command || exit 2
		./FiVoNAGI || exit 3
		echo "$(date +"%x %T") ::end    ::$execname" 
		anglek=$[anglek + 1]
	    done
	done
    done
done
