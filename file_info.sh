#!/bin/bash

for loopdirectory in `ls -d *`; do
    #echo $loopdirectory;
    for datadirectory in `ls $loopdirectory/*`; do 
        echo $datadirectory;
        soxi "$datadirectory";
    done
done 



                             
