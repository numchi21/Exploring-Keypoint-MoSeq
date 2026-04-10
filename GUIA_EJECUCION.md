# Guía de ejecución en GCP

## Requisitos previos
- VM `kpms-explore` creada y corriendo (`g2-standard-4` + L4, Ubuntu Deep Learning)
- Imagen Docker buildeada y en Artifact Registry
- Datos `.h5` subidos a la VM

---

## 1. Comprobar que la VM está corriendo

```bash
gcloud compute instances list --project=kpms-tfm-try
```

Si no está corriendo, arráncala:

```bash
gcloud compute instances start kpms-explore --zone=us-east1-b --project=kpms-tfm-try
```

Obtén la IP:

```bash
gcloud compute instances list --project=kpms-tfm-try --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
```

---

## 2. Si has cambiado código (01_explore.py, 01_config.py, etc.)

Desde tu Mac, sube los ficheros modificados:

```bash
scp Explore/01_explore.py Explore/01_config.py numchi2121@<IP>:~/explore/Explore/
scp Model/config.py numchi2121@<IP>:~/explore/Model/
```

Conéctate a la VM y rebuildea:

```bash
ssh numchi2121@<IP>
docker build -t europe-west1-docker.pkg.dev/kpms-tfm-try/kpms/explore ~/explore
```

---

## 3. Lanzar el contenedor

Desde la VM:

```bash
docker run -it --gpus all \
  -v ~/data:/app/data \
  -v ~/kpms_project:/app/kpms_project \
  europe-west1-docker.pkg.dev/kpms-tfm-try/kpms/explore
```

---

## 4. Ejecutar el script

Dentro del contenedor:

```bash
PYTHONPATH=/app python3.10 Explore/01_explore.py
```

---

## 5. Apagar la VM cuando termines (importante para no gastar crédito)

```bash
gcloud compute instances stop kpms-explore --zone=us-east1-b --project=kpms-tfm-try
```

---

## Notas importantes

- Los resultados se guardan en `~/kpms_project` en la VM (montado como `/app/kpms_project` en el contenedor)
- Si la VM no tiene GPU disponible en la zona, usa el script `terraform/retry_vm.sh` para buscar otra zona
- El `DATA_ROOT` en `Model/config.py` debe ser `Path("data/h5")` (sin `../`)
- Usar siempre `python3.10` dentro del contenedor, no `python`

## 6. Ver plots sin parar la ejecución

Los plots se guardan en `~/kpms_project/QA_AUDIT/` en la VM. Descárgalos desde tu Mac sin interrumpir nada:

```bash
scp numchi2121@35.227.52.54:~/kpms_project/QA_AUDIT/pca_variance.png . && open pca_variance.png
```

Para descargar todos los plots de golpe:

```bash
scp -r numchi2121@35.227.52.54:~/kpms_project/QA_AUDIT/ ./QA_AUDIT_remote/
```
