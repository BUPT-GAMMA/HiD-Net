#!/bin/bash
# execute: ./parallel.sh 2
# 2 means gpu
if [ ! -n "$1" ]; then
  echo "gpu is empty"
  exit 0
fi
if [ ! -n "$2" ]; then
  echo "repeat is empty"
  exit 0
fi
if [ ! -n "$3" ]; then
  echo "dataset is empty"
  exit 0
fi
if [ ! -n "$4" ]; then
  echo "cal is empty"
  exit 0
fi
if [ ! -n "$5" ]; then
  echo "config file is empty"
  exit 0
fi
if [ ! -n "$6" ]; then
  echo "predict file is empty"
  exit 0
fi
if [ ! -n "$7" ]; then
  echo "start is empty"
  exit 0
fi
if [ ! -n "$8" ]; then
  echo "end is empty"
  exit 0
fi
# datasets=(cora citeseer pubmed chameleon actor squirrel)
datasets=(cora citeseer chameleon actor squirrel)
#datasets=(chameleon actor squirrel)
for i in `seq $7 $8`; do
  echo "===================================================================================="
  file="prediction/config/$5/${i}.yaml"
  if [ ! -f ${file} ]
  then
    echo "yaml not exists"
  fi
  for dataset in ${datasets[*]}; do
    {
    excel="prediction/excel/$6/${dataset}_$4_${i}.csv"
    if [ ! -f ${excel} ]
    then
      para="--tune True --earlystop True --configfile $5 --predictfile $6 --times ${i} --repeat $2 --dataset ${dataset} --calg $4 --gpu $1"
      echo "***************************************************************************************"
      echo ${para}
      python train.py ${para}
    else
      echo ${excel} exists
    fi
    }

  done
  wait
done


