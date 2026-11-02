package synth

// Surface "seeds": each batch draws a few of these so generations vary in domain, tools, length and style.

const systemPrompt = `You write synthetic data for a machine-learning benchmark. Follow the per-item specifications exactly ` +
	`(each item's assigned label is binding: it is the ground truth). Make items realistic and varied, never templated: ` +
	`vary names, numbers, wording, formatting and length. Never use real credentials or real personal data. ` +
	`Reply with one JSON object of the form {"items": [...]} and nothing else.`

var projects = []string{
	"a Django e-commerce backend", "a Rust CLI that parses server logs", "a React Native fitness app",
	"a Go payments microservice", "an Airflow data pipeline", "a Terraform AWS landing zone", "a Unity mobile game",
	"a Kubernetes operator in Go", "a PyTorch model-training repo", "a Rails SaaS billing app",
	"a Node.js GraphQL API", "embedded C firmware for a thermostat", "a Flutter delivery app",
	"a Spring Boot banking service", "a Hugo documentation site", "a Jupyter analytics repo", "a Chrome extension",
	"an Elixir Phoenix chat app", "a Haskell type checker", "a Swift iOS photo editor", "a Laravel CMS",
	"a Fortran climate model", "a dotfiles repo of shell scripts", "a Kotlin Android banking app",
	"a .NET inventory system", "a Postgres migration toolkit", "a Scala Spark ETL job", "a SvelteKit dashboard",
	"a C++ physics engine", "a Python package published to PyPI", "a Next.js marketing site",
	"a FastAPI ML inference server", "a Deno edge function", "a Zig HTTP server", "an Ansible playbook repo",
	"a COBOL batch job being modernized", "a Vue admin panel", "a gRPC streaming service in Java",
	"a WordPress plugin", "a Bazel monorepo", "a home-automation Python daemon", "a Solidity smart-contract repo",
	"a static-analysis tool in OCaml", "a LaTeX thesis repo with build scripts", "an R package for statistics",
}

var lengths = []string{
	"very short (1-2 lines)", "short (3-6 lines)", "medium (8-15 lines)", "long (20-35 lines)",
}

var industries = []string{
	"airline", "retail bank", "mobile carrier", "home internet provider", "food delivery app", "online clothing store",
	"SaaS project-management tool", "car rental company", "hotel chain", "electric utility", "health insurer",
	"video streaming service", "gym chain", "furniture retailer", "ride-hailing app", "university IT helpdesk",
	"accounting software", "pet supplies shop", "concert ticketing site", "cloud hosting provider",
	"game publisher", "online pharmacy", "language-learning app", "smart-home device maker", "moving company",
	"dental clinic", "crypto exchange", "public library", "parcel courier", "meal-kit subscription",
}

var channels = []string{"email", "live chat", "app store review", "support ticket form", "social media post",
	"SMS", "phone call transcript summary", "community forum post"}

var writerStyles = []string{
	"formal and complete sentences", "casual, lowercase, little punctuation", "non-native English speaker",
	"terse, few words", "rambling with backstory", "uses bullet points", "polite but direct", "sarcastic",
	"lots of specific details (order numbers, dates, amounts)", "older customer, very formal",
}

var topics = []string{
	"astronomy", "medieval history", "cooking and food chemistry", "personal finance", "marine biology",
	"software engineering", "football (soccer) tactics", "classical music", "volcanology", "urban planning",
	"nutrition science", "aviation", "chess", "renaissance art", "climate science", "law and contracts",
	"genetics", "photography", "ancient Rome", "machine learning", "gardening", "linguistics", "economics",
	"world geography", "the human immune system", "electric vehicles", "mythology", "architecture",
	"mountaineering", "pharmacology", "cryptography", "film history", "beekeeping", "oceanography",
	"tax rules for freelancers", "wine making", "robotics", "public health", "the history of printing", "coffee",
}

var logSystems = []string{
	"a Java Spring Boot API server", "an nginx reverse proxy access log", "a GitHub Actions CI build",
	"Kubernetes pod events and kubelet messages", "a PostgreSQL server log", "a systemd journal on an Ubuntu host",
	"docker compose up output for a web stack", "a Python Celery worker", "a Node.js Express app",
	"a Terraform apply run", "an Airflow DAG run", "a Redis server log", "a cron job running nightly backups",
	"an Android app logcat", "a Go gRPC service with structured JSON logs", "a Jenkins pipeline",
	"an HAProxy load balancer", "a Kafka consumer group", "an IoT gateway", "a Rails app production log",
	"an Elasticsearch node", "a Windows service event log export", "a Prometheus alertmanager",
	"a game server", "a payment reconciliation batch job",
}

var deployTools = []string{
	"GitHub Actions", "GitLab CI", "Jenkins", "CircleCI", "Argo CD", "Spinnaker", "Buildkite",
	"a custom bash deploy script", "Ansible", "Helm via a Makefile", "AWS CodePipeline", "Heroku release phase",
	"Fly.io deploy", "Capistrano", "Terraform Cloud", "Octopus Deploy",
}
