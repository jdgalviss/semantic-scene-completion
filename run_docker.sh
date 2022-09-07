#!/bin/bash
sudo xhost +si:localuser:root
XSOCK=/tmp/.X11-unix

docker run -it --rm  \
    --net=host  \
    --privileged \
    --runtime=nvidia \
    -e DISPLAY=$DISPLAY \
    -v $XSOCK:$XSOCK \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v `pwd`/semantic-scene-completion:/usr/src/app/semantic-scene-completion \
    -v /media/jdgalviss/JDG/TUM/ssc/data/semantickitti/dataset/dataset/:/usr/src/app/data \
    --shm-size 8G \
    ssc "$@"