#!/bin/bash

# LOOP FROM 1 TO 512
for i in {1..33280}
do
    # CREATE A FILE NAMED final_weight_$i.txt
    touch "weight_${i}.txt"

    # GENERATE A RANDOM FLOATING-POINT VALUE BETWEEN -1 AND 1
    random_value=$(awk -v seed="$RANDOM" 'BEGIN { srand(seed); printf "%.6f", (rand() * 2) - 1 }')

    # WRITE THE RANDOM FLOATING-POINT VALUE TO THE FILE
    echo "$random_value" > "weight_${i}.txt"
    
    echo "file weight_${i}.txt created with value $random_value"
done

