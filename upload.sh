#!/bin/bash
# Corre en LOCAL: sube código y datos a la VM

VM_IP="35.227.52.54"
VM_USER="numchi2121"
SSH="ssh -o StrictHostKeyChecking=no $VM_USER@$VM_IP"
SCP="scp -o StrictHostKeyChecking=no"

echo "==> Creando directorios en la VM..."
$SSH "mkdir -p ~/explore ~/data/h5"

echo "==> Subiendo scripts de Explore/..."
$SCP Explore/01_explore.py Explore/01_config.py Explore/Dockerfile Explore/requirements.txt $VM_USER@$VM_IP:~/explore/

echo "==> Subiendo datos h5..."
$SCP -r data/h5/ $VM_USER@$VM_IP:~/data/

echo "==> Listo. Conéctate con: ssh $VM_USER@$VM_IP"
