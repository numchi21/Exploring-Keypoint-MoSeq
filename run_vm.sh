#!/bin/bash
# Corre en la VM: buildea imagen y abre shell lista para ejecutar

REGISTRY="europe-west1-docker.pkg.dev/kpms-tfm-try/kpms/explore"

echo "==> Buildeando imagen..."
docker build -t $REGISTRY ~/explore

echo "==> Listo. Abriendo shell en el contenedor..."
docker run -it --gpus all \
  -v ~/data:/app/data \
  -v ~/kpms_project:/app/kpms_project \
  $REGISTRY bash
