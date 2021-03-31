#!/bin/bash

rq worker &>/dev/null &
worker1_pid=$!

rq worker &>/dev/null &
worker2_pid=$!

rq worker &>/dev/null &
worker3_pid=$!

rq worker &>/dev/null &
worker4_pid=$!

rq worker &>/dev/null &
worker5_pid=$!

trap onexit INT
function onexit() {
  kill -9 $worker1_pid
  kill -9 $worker2_pid
  kill -9 $worker3_pid
  kill -9 $worker4_pid
  kill -9 $worker5_pid
}

gunicorn --bind 0.0.0.0:5000 wsgi:app