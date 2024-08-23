#!/bin/bash

# LOOP FROM 1 TO 262144

for i in {1..1024}
do

    # CREATE A FILE NAMED semi_final_weight$i.txt
    touch "final_weight_${i}.txt"
    
    echo "file ${i} created"
done
