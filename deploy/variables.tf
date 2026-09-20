variable "project_name" {
  description = "Name of the Railway project."
  type        = string
  default     = "worldgen-campaign"
}

variable "source_repo" {
  description = "GitHub repository Railway builds from, as owner/name."
  type        = string
  default     = "countercheck/worldgen"
}

variable "source_repo_branch" {
  description = "Branch Railway deploys. A push to it is a deploy."
  type        = string
  default     = "master"
}

variable "region" {
  description = "Railway region. Put it near the players, not near you: the referee's clicks are rare and the players' map redraws are not."
  type        = string
  default     = "us-east4-eqdc4a"
}

variable "subdomain" {
  description = "Subdomain under Railway's domain. The join links are built from whatever this resolves to."
  type        = string
  default     = "worldgen-campaign"
}

variable "body_limit_bytes" {
  description = "Largest accepted request body. This is the world upload limit: a 32x32 world is 641 KB and a 64x64 about 2.5 MB."
  type        = number
  default     = 16777216

  validation {
    # Fastify's own default is 1 MiB, which is below the world the documentation tells a
    # referee to generate. Anything at or under it reintroduces the 413 this replaced.
    condition     = var.body_limit_bytes > 1048576
    error_message = "A limit of 1 MiB or less refuses the 64x64 world docs/CAMPAIGN.md tells you to generate."
  }
}

variable "rate_limit_global" {
  description = "Requests per minute per address, across everything."
  type        = number
  default     = 600
}

variable "rate_limit_create" {
  description = "Campaign creations per minute per address. The only unauthenticated write."
  type        = number
  default     = 5
}
