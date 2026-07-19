#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "fake-lkh-1.0"
  exit 0
fi
cp init.tour out.tour
exit 0
