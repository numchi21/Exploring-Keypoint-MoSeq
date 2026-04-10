variable "project_id" {
  description = "GCP project ID"
  type        = string
  default = "kpms-tfm-try"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-east1"  # Amsterdam que normalmente tiene mas que madrid que sería 4
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-east1-b" #da igual
}

variable "ssh_user" {
  description = "SSH username"
  type        = string
  default = "numchi2121"
}

variable "ssh_pub_key_path" {
  description = "Path to SSH public key"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}
