#!/bin/bash
#

ps=(1 2)
etas=(20 41 82)
ispams=(30 20 10)  
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
	if [ "$etai" == 20 ]
	then
	    etain=1
	elif [ "$etai" == 41 ]
	then
	    etain=2
	elif [ "$etai" == 82 ]
	then
	    etain=4
	fi
	for cfli in "${cfls[@]}"
	do
	    for i in "${ispams[@]}"
	    do
		execname="eta${etai}-cfl${cfli}-p${pi}-a${i}"
		if [ ${EXPORT} ]
		then
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
		command="nvcc -D ROOT=${ROOT} -D DEPLOY=${deploy_path} ${EXPORT_STR} -D PRECISION=$pi -D ISPAM=$i -D CFLW=$cfli -D ETAC=$etain -o FiVoNAGI -lcuda -lcudart  -lm -lGL -lGLU -lglut -lpthread -arch=sm_13  ./driver.cu"
		echo $command
		eval $command || exit 2
		./FiVoNAGI || exit 3
		echo "$(date +"%x %T") ::end    ::$execname"

		# the following is a quick and dirty patch
		if [ ${EXPORT} ]
		then 
		    mv "${deploy_path}/u1max.dat" "${deploy_path}/${execname}.dat"
		fi
	    done
	done
    done
done
