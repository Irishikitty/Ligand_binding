#!/bin/bash
start=1
end=2
dir="/data/ml20gd/Ligand_binding/checkpoints/simple_generator/"

for seed in 2000; do
  for iterations in 0; do
      for ((epoch=$start; epoch<=$end; epoch+=3));do
        file=$dir$epoch"_net_G.pth"
        echo $file
        if [ -f "$file" ]; then
                echo $file "exists"
                python3 test.py --epoch=$epoch --seed $seed --iterations $iterations
        fi
    done
  done
done

