#!/bin/bash  

configs=(
    "configs/UniGOOD_configs/GDMotif/UniGOOD_pretrain.yaml"
    "configs/UniGOOD_configs/GDMotif/UniGOOD.yaml"

    "configs/UniGOOD_configs/GDTox21SIDER/UniGOOD_pretrain.yaml"
    "configs/UniGOOD_configs/GDTox21SIDER/UniGOOD.yaml"

    "configs/UniGOOD_configs/GDBBBPBACE/UniGOOD_pretrain.yaml"
    "configs/UniGOOD_configs/GDBBBPBACE/UniGOOD.yaml"

    "configs/UniGOOD_configs/GDHIVZINC/UniGOOD_pretrain.yaml"
    "configs/UniGOOD_configs/GDHIVZINC/UniGOOD.yaml"
)  

rounds=(1 2 3)
for round in "${rounds[@]}"; do
    for cfg in "${configs[@]}"; do
        goodtg --config_path "$cfg" --exp_round "$round" --gpu_idx 1 &  
        PID=$!  
        wait $PID
    done
done  