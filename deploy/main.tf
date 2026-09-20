# The campaign server, on Railway.
#
#     cd deploy
#     export RAILWAY_TOKEN=...        # an account token, not a project token
#     tofu init
#     tofu apply
#
# What this describes is one service, one volume, one replica. That is not modesty about
# the tooling — it is the shape of the application. The campaign is a SQLite file which
# only one process may write, and the WebSocket listeners that push a view to each player
# are held in that process's memory. A second replica is not more capacity; it is a second
# campaign that half the players are connected to.
#
# Railway builds the image itself from the connected repository, using `railway.json` at
# the root to find `campaign/Dockerfile`. The build context is the repository root because
# the client imports a world fixture from `shared/test/fixtures` — see the Dockerfile.

terraform {
  required_version = ">= 1.6"

  required_providers {
    railway = {
      source  = "terraform-community-providers/railway"
      version = "~> 0.6"
    }
  }
}

# Reads `RAILWAY_TOKEN` from the environment. Deliberately not written into a file here:
# a token in a tfvars is a token in a backup, in an editor's recent list, and eventually
# in a repository.
provider "railway" {}

resource "railway_project" "campaign" {
  name        = var.project_name
  description = "Napoleonic campaign server"

  # The default. Stated rather than assumed, because the join links are the whole security
  # model and a project that quietly became public would put the console behind nothing.
  private = true

  # No per-pull-request environments. Each would want its own volume, and a throwaway
  # environment holding a copy of somebody's campaign is not a thing to create by accident.
  has_pr_deploys = false
}

resource "railway_service" "campaign" {
  name       = "campaign"
  project_id = railway_project.campaign.id

  source_repo        = var.source_repo
  source_repo_branch = var.source_repo_branch

  # The volume is the campaign. `CAMPAIGN_DB` below points inside it, and the image
  # already declares `/data` — this is what makes that directory outlive a deploy.
  volume = {
    name       = "campaign-data"
    mount_path = "/data"
  }

  # One. See the note at the top of this file; this is the line that enforces it.
  regions = [
    {
      region       = var.region
      num_replicas = 1
    }
  ]
}

resource "railway_service_domain" "campaign" {
  environment_id = railway_project.campaign.default_environment.id
  service_id     = railway_service.campaign.id
  subdomain      = var.subdomain
}

# ---- what the process reads ------------------------------------------------
#
# The image sets sensible defaults for PORT, HOST, CAMPAIGN_DB and CAMPAIGN_CLIENT. What
# is set here is only what the image cannot know: that it is behind a router, and what its
# budgets are.

locals {
  environment = {
    # Railway terminates TLS and forwards, so the connecting address is the router's.
    # Without this every player shares one rate-limit bucket and the first one spends it.
    # `true` rather than a list of addresses because the router is the only way in.
    TRUST_PROXY = "true"

    # Stated rather than left to the image, so that raising it is an edit here and not a
    # remembered console click. A 64x64 world is about 2.5 MB; this clears 128x128.
    CAMPAIGN_BODY_LIMIT = tostring(var.body_limit_bytes)

    # Per address, per minute. Generous for a referee mid-evening, mean about creation —
    # the one unauthenticated write, and the one that puts a world on the volume.
    CAMPAIGN_RATE_GLOBAL = tostring(var.rate_limit_global)
    CAMPAIGN_RATE_CREATE = tostring(var.rate_limit_create)
  }
}

resource "railway_variable" "campaign" {
  for_each = local.environment

  environment_id = railway_project.campaign.default_environment.id
  service_id     = railway_service.campaign.id
  name           = each.key
  value          = each.value
}
