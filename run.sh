#!/bin/bash
# usage: run.sh camera or run.sh video
set -e

cd bin
rm -rf *
cd ../build
rm -rf *
cmake ..
make -j12
#make
cd ../
#./bin/buildEngine
#./bin/runDet
./bin/pose_test
