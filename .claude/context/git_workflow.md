# Git Workflow & Standards

## Branch Naming

All branches must follow this pattern:

```
<type>/<author>/<short-description>
```

### Fields

- **`<author>`** — Your name in lowercase (e.g., `lucas`, `alex`)
- **`<type>`** — One of the type prefixes below
- **`<short-description>`** — Lowercase, hyphen-separated, max ~4 words

### Type Prefixes

| Type | When to use |
|------|------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `data` | Data pipeline changes (scripts, manifests, DVC stages) |
| `model` | Model architecture, training, or checkpoint changes |
| `infra` | Docker, CI, server config changes |
| `refactor` | Code restructure without behaviour change |
| `docs` | Documentation only |
| `exp` | Throwaway experiments (will not be merged to main) |

### Examples

```
feat/lucas/ambient-retrieval-endpoint
fix/lucas/vocoder-resampling-bug
data/alex/birdnet-annotation-index
model/lucas/vae-beta-annealing
infra/alex/docker-compose-ai-server
exp/lucas/latent-diffusion-prototype
```

---

## Commit Messages

Use the **imperative mood, present tense**. Keep the subject line under 72 characters.

### Examples

```
Add ambient retrieval function to inference.py
Fix days_since_rain UTC/AEST off-by-one error
Train ecoacoustic HiFi-GAN on site 257 clips
Update docker-compose to expose AI server port
```

### Rules

- **Do NOT** reference issue numbers or internal task IDs in the subject line — put those in the body if needed
- Subject line should be a single, clear statement of what the commit does
- Use imperative tense: "Add", "Fix", "Update", not "Added", "Fixed", "Updated"

---

## Author

Lucas Tao

**Last updated:** 2026-05-01
