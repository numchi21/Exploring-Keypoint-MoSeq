terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ── Enable APIs ───────────────────────────────────────────────────────────────
resource "google_project_service" "compute" {
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

# ── Artifact Registry ────────────────────────────────────────────────────────
resource "google_artifact_registry_repository" "kpms" {
  depends_on    = [google_project_service.artifactregistry]
  repository_id = "kpms"
  format        = "DOCKER"
  location      = var.region
  description   = "Keypoint-MoSeq Docker images"
}

# ── Service Account for the VM ───────────────────────────────────────────────
resource "google_service_account" "vm_sa" {
  account_id   = "kpms-explore-vm"
  display_name = "Keypoint-MoSeq Explore VM"
}

resource "google_project_iam_member" "vm_sa_registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.vm_sa.email}"
}

# ── Compute Engine VM ────────────────────────────────────────────────────────
resource "google_compute_instance" "explore" {
  depends_on   = [google_project_service.compute]
  name         = "kpms-explore"
  machine_type = "a2-ultragpu-1g"

  boot_disk {
    initialize_params {
      image = "projects/deeplearning-platform-release/global/images/family/common-cu128-ubuntu-2204-nvidia-570"
      size  = 100  # GB — espacio para imagen Docker + datos
      type  = "pd-ssd"
    }
  }

  network_interface {
    network = "default"
    access_config {}  # IP pública para SSH
  }

  guest_accelerator {
    type  = "nvidia-a100-80gb"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"  # requerido con GPU
    automatic_restart   = false
  }

  service_account {
    email  = google_service_account.vm_sa.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    ssh-keys = "${var.ssh_user}:${file(pathexpand(var.ssh_pub_key_path))}"
  }

  tags = ["kpms-explore"]
}

# ── Firewall: allow SSH ───────────────────────────────────────────────────────
resource "google_compute_firewall" "allow_ssh" {
  depends_on = [google_project_service.compute]
  name    = "kpms-allow-ssh"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  target_tags   = ["kpms-explore"]
  source_ranges = ["0.0.0.0/0"]
}

locals {
  image_url = "${var.region}-docker.pkg.dev/${var.project_id}/kpms/explore"
}

# ── Build & Push imagen ───────────────────────────────────────────────────────
resource "null_resource" "build_and_push" {
  depends_on = [google_artifact_registry_repository.kpms]

  triggers = {
    # Re-ejecuta si cambia cualquier fichero de Explore/
    dockerfile  = filemd5("${path.module}/../Explore/Dockerfile")
    explore_py  = filemd5("${path.module}/../Explore/01_explore.py")
    config_py   = filemd5("${path.module}/../Explore/01_config.py")
    requirements = filemd5("${path.module}/../Explore/requirements.txt")
  }

  provisioner "local-exec" {
    command = <<-EOT
      gcloud auth configure-docker ${var.region}-docker.pkg.dev --quiet
      docker build -t ${local.image_url} ${path.module}/../Explore
      docker push ${local.image_url}
    EOT
  }
}

# ── Deploy en VM ──────────────────────────────────────────────────────────────
resource "null_resource" "deploy_to_vm" {
  depends_on = [null_resource.build_and_push, google_compute_instance.explore]

  triggers = {
    build_id = null_resource.build_and_push.id
  }

  provisioner "remote-exec" {
    connection {
      type        = "ssh"
      host        = google_compute_instance.explore.network_interface[0].access_config[0].nat_ip
      user        = var.ssh_user
      private_key = file(pathexpand(replace(var.ssh_pub_key_path, ".pub", "")))
    }

    inline = [
      "docker-credential-gcr configure-docker --registries=${var.region}-docker.pkg.dev",
      "docker pull ${local.image_url}",
    ]
  }
}
