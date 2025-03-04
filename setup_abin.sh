#!/bin/bash

answer=`which bmri-prep`
if [[ -z ${answer} ]];
then
    echo "bmri-prep not in PATH, adding ..."
    currentpath=`pwd`
    echo "# bmri-prep" >> ~/.bashrc
    echo 'PATH=$PATH:'"${currentpath}/abin" >> ~/.bashrc
    echo "... done. Please restart your terminal for changes to take effect."
else
    echo "bmri_prep already in PATH, nothing to be done."
fi


