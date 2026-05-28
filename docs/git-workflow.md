# Git Workflow & Branching Strategy

This document describes the Git workflow and branching strategy for the VLA-LEGO project.

## Branch Structure

```
main                    # Production-ready code
├── develop             # Integration branch (optional)
└── feature/*           # Feature branches
    ├── feature/data-pipeline
    ├── feature/transformer-v2
    └── feature/deepspeed-integration
```

### Branch Types

| Branch | Purpose | Merges To |
|--------|---------|-----------|
| `main` | Stable, tested code. All CI must pass. | - |
| `develop` | Integration branch (optional) | `main` |
| `feature/*` | New features | `main` or `develop` |
| `fix/*` | Bug fixes | `main` |
| `exp/*` | Experiments (may not merge) | `feature/*` or delete |

## Workflow

### Starting a New Feature

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/my-feature

# Work on feature...
git add .
git commit -m "feat: add new feature"

# Push to remote
git push -u origin feature/my-feature
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(model): add SwiGLU activation

fix(data): handle empty sequences in dataloader

docs: update README with cluster instructions

refactor(train): extract checkpoint logic to separate module
```

### Pull Request Process

1. **Create PR** from feature branch to `main`
2. **CI runs automatically** (lint, tests, smoke test)
3. **Request review** from team member
4. **Address feedback**
5. **Squash and merge** when approved

### Merge Strategy

- **Squash merge** for feature branches (cleaner history)
- **Regular merge** for releases/hotfixes (preserve history)

## Milestone Tags

Tags mark significant project milestones (per roadmap):

```bash
# Phase 0 complete
git tag -a v0.1.0 -m "Phase 0: Foundation & Setup complete"
git push origin v0.1.0

# Phase 1 complete
git tag -a v0.2.0 -m "Phase 1: Data Pipeline complete"
git push origin v0.2.0
```

### Tag Naming Convention

```
v<major>.<minor>.<patch>

v0.1.0  - Phase 0 complete
v0.2.0  - Phase 1 complete
v0.3.0  - Phase 2 complete
v1.0.0  - First working model
```

## Protected Branch Rules

Configure in GitHub Settings → Branches:

### `main` branch protection:
- [x] Require pull request reviews (1 reviewer)
- [x] Require status checks to pass:
  - `lint`
  - `test`
  - `smoke`
- [x] Require branches to be up to date
- [x] Do not allow bypassing

## Quick Reference

```bash
# Start feature
git checkout main && git pull
git checkout -b feature/xyz

# Save work
git add . && git commit -m "feat: description"
git push -u origin feature/xyz

# Update from main
git fetch origin
git rebase origin/main
# or: git merge origin/main

# Clean up after merge
git checkout main && git pull
git branch -d feature/xyz

# Tag release
git tag -a v0.1.0 -m "Description"
git push origin v0.1.0
```

## CI/CD Integration

Every push/PR triggers:
1. **Lint** - Code formatting checks
2. **Type Check** - MyPy static analysis
3. **Unit Tests** - Fast tests (<2 min)
4. **Smoke Test** - Import and config validation

Every merge to `main` triggers:
1. All CI checks
2. **Container Build** - Docker image
3. **Apptainer Build** - HPC container
