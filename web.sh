#!/bin/bash

JARFILE=web
PID_FILE=/home/hyper/projects/captchaCracker/dist/web/pid.file
RUNNING=N
PWD=`pwd`

######### DO NOT MODIFY ########

if [ -f $PID_FILE ]; then
        PID=`cat $PID_FILE`
        if [ ! -z "$PID" ] && kill -0 $PID 2>/dev/null; then
                RUNNING=Y
        fi
fi

start()
{
        if [ $RUNNING == "Y" ]; then
                echo "Application already started"
        else
                if [ -z "$JARFILE" ]
                then
                        echo "ERROR: exec file not found"
                else
                        cd /home/hyper/projects/captchaCracker/dist/web/
                        nohup ./$JARFILE > web.log 2>&1  &
                        echo $! > $PID_FILE
                        echo "Application $JARFILE starting..."
                        tail -f web.log
                fi
        fi
}

stop()
{
        if [ $RUNNING == "Y" ]; then
                kill -9 $PID
                rm -f $PID_FILE
                echo "Application stopped"
        else
                echo "Application not running"
        fi
}

restart()
{
        stop
        start
}

case "$1" in

        'start')
                start
                ;;

        'stop')
                stop
                ;;

        'restart')
                restart
                ;;

        *)
                echo "Usage: $0 {  start | stop | restart  }"
                exit 1
                ;;
esac
exit 0



