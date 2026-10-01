import { defineRailway, github, preserve, project, service, volume } from "railway/iac";

// The campaign server, one container, and the volume its database lives on; and the world
// generator's web interface beside it.
//
// Railway does not read this file on deploy. A change here reaches the service only when
// `railway config apply` is run, and `railway config plan` shows what that would change.

export default defineRailway(() => {
  // The campaign's SQLite database. `campaign/Dockerfile` points CAMPAIGN_DB inside it.
  const campaignData = volume("campaign-data", {
    alerts: { usage: { "100": {}, "80": {}, "95": {} } },
    allowOnlineResize: true,
    region: "us-east4-eqdc4a",
    sizeMB: 5000,
  });

  const campaign = service("campaign", {
    source: github("countercheck/worldgen", { branch: "master", checkSuites: false }),
    build: { buildEnvironment: "V3", builder: "DOCKERFILE", dockerfilePath: "campaign/Dockerfile" },
    healthcheck: "/health",
    healthcheckTimeout: 30,
    // One replica: SQLite on one volume has exactly one writer.
    replicas: { "us-east4-eqdc4a": 1 },
    deploy: {
      drainingSeconds: 0,
      overlapSeconds: 0,
      restartPolicyType: "ON_FAILURE",
      restartPolicyMaxRetries: 10,
      sleepApplication: false,
      // Refuse to deploy without the volume, rather than start on a disk that forgets.
      requiredMountPath: "/data",
    },
    volumeMounts: { "/data": campaignData },
    networking: { serviceDomains: { "worldgen-campaign.up.railway.app": {} } },
    // What the image cannot know for itself. PORT, HOST, CAMPAIGN_DB and CAMPAIGN_CLIENT
    // are set in the image and left alone here.
    env: {
      // Railway terminates TLS and forwards, so the connecting address is the router's.
      // Without this every player shares one rate-limit bucket and the first one spends it.
      TRUST_PROXY: "true",
      // The world upload limit, 64 MiB. A 200x200 world is 32 MB as generated, and
      // anything at or under 32 MiB refuses it.
      CAMPAIGN_BODY_LIMIT: "67108864",
      // Per address, per minute: generous for a referee mid-evening, mean about creation,
      // the one unauthenticated write and the one that puts a world on the volume.
      CAMPAIGN_RATE_GLOBAL: "600",
      CAMPAIGN_RATE_CREATE: "5",
    },
  });

  // `worldgen serve`: the generator behind a password. See `deploy/generator.Dockerfile`.
  const generator = service("generator", {
    source: github("countercheck/worldgen", { branch: "master", checkSuites: false }),
    build: {
      buildEnvironment: "V3",
      builder: "DOCKERFILE",
      dockerfilePath: "deploy/generator.Dockerfile",
      // Rebuilt for a change to the generator, not for every campaign commit.
      watchPatterns: ["worldgen/**", "pyproject.toml", "deploy/generator.Dockerfile*"],
    },
    healthcheck: "/health",
    healthcheckTimeout: 30,
    // One replica: a generated world and the stream reporting its progress live in one
    // process's memory, and a request routed to a second replica would find neither.
    replicas: { "us-east4-eqdc4a": 1 },
    deploy: {
      restartPolicyType: "ON_FAILURE",
      restartPolicyMaxRetries: 10,
      // Asleep when nobody is using it. A run in progress keeps it awake — its progress
      // stream sends every 15 seconds — and what sleep costs is the worlds held in memory,
      // which a user downloads rather than keeps there.
      sleepApplication: true,
    },
    networking: { serviceDomains: { "worldgen-generator.up.railway.app": {} } },
    env: {
      // Set in Railway, never here. `preserve()` keeps whatever value the service holds,
      // and without one `worldgen serve` refuses to start rather than serve it open.
      WORLDGEN_PASSWORD: preserve(),
    },
  });

  return project("worldgen-campaign", {
    resources: [campaign, campaignData, generator],
  });
});
