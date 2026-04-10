output "vm_external_ip" {
  description = "IP pública de la VM para SSH"
  value       = google_compute_instance.explore.network_interface[0].access_config[0].nat_ip
}

output "artifact_registry_url" {
  description = "URL del registry para docker push/pull"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/kpms"
}

output "ssh_command" {
  description = "Comando SSH para conectarse"
  value       = "ssh ${var.ssh_user}@${google_compute_instance.explore.network_interface[0].access_config[0].nat_ip}"
}
