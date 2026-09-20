output "url" {
  description = "Where the referee and the players go. Join links are this plus #/j/<campaign>/<token>."
  value       = "https://${railway_service_domain.campaign.domain}"
}

output "project_id" {
  description = "For `railway link`, and for finding the volume in the console when a backup is wanted."
  value       = railway_project.campaign.id
}

output "service_id" {
  value = railway_service.campaign.id
}
