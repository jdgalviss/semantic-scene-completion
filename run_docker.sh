#!/bin/bash
# sudo xhost +si:localuser:root
# XSOCK=/tmp/.X11-unix

sudo docker run -it --rm  \
    --net=host  \
    --privileged \
    --runtime=nvidia \
    -v `pwd`/semantic-scene-completion:/usr/stud/gaj/ssc/semantic-scene-completion/semantic-scene-completion \
    -v /home/galvis/data/ssc_dataset/dataset/:/storage/user/gaj/ssc/dataset \
    --shm-size 32G \
    ssc "$@"

    # -v /home/galvis/data/semantickitti/dataset/dataset/:/storage/user/gaj/ssc/dataset \

    # -e DISPLAY=$DISPLAY \
    # -v $XSOCK:$XSOCK \
    # -v $HOME/.Xauthority:/root/.Xauthority \