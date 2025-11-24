# GitHub Actions Complete Guide

Learn how to automate your workflows with GitHub Actions - from understanding the basics to writing your own custom workflows.

## What Are GitHub Actions?

**GitHub Actions** is a CI/CD (Continuous Integration/Continuous Deployment) platform that allows you to automate tasks in your repository.

### Why Use GitHub Actions?

**Common use cases:**
- 🚀 **Deploy websites** automatically when you push code
- 🧪 **Run tests** on every pull request
- 📦 **Build and publish** packages
- 🔄 **Sync data** from external sources (like Confluence)
- 🤖 **Automate repetitive tasks** (labeling issues, formatting code)

**Think of it as:** A robot assistant that watches your repository and does tasks automatically when things happen.

### How It Works

```
You push code to GitHub
         ↓
GitHub detects the push
         ↓
Workflow file (.yml) is triggered
         ↓
GitHub spins up a virtual machine
         ↓
Runs your commands step by step
         ↓
Reports success or failure
```

**The magic:** It's like having a server that runs commands for you, but you don't have to manage or pay for the server!

---

## Anatomy of a Workflow File

All GitHub Actions workflows live in `.github/workflows/` and use YAML format.

### Basic Structure

```yaml
name: Workflow Name              # 1. What shows up in GitHub UI

on:                              # 2. WHEN to run this workflow
  push:
    branches: [main]

jobs:                            # 3. WHAT to do (can have multiple jobs)
  job-name:
    runs-on: ubuntu-latest       # 4. WHERE to run (virtual machine)
    steps:                       # 5. Step-by-step instructions
      - name: Step description
        run: echo "Hello!"
```

### The 5 Key Sections Explained

#### 1. `name` - Workflow Name

```yaml
name: Deploy to GitHub Pages
```

**What it does:** Labels your workflow in the GitHub Actions tab

**Why it matters:** Helps you identify workflows when you have multiple

**Tip:** Use descriptive names like "Deploy Production" or "Run Tests"

---

#### 2. `on` - Triggers (When to Run)

This is the most important part - it defines WHEN your workflow runs.

**Common triggers:**

```yaml
# Run on push to specific branches
on:
  push:
    branches:
      - main
      - develop

# Run on pull requests
on:
  pull_request:
    branches: [main]

# Run on schedule (cron)
on:
  schedule:
    - cron: '0 6 * * *'  # Every day at 6 AM UTC

# Run manually
on:
  workflow_dispatch:

# Multiple triggers
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
```

**Understanding cron syntax:**
```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday to Saturday)
│ │ │ │ │
* * * * *
```

Examples:
- `0 0 * * *` = Every day at midnight
- `0 */6 * * *` = Every 6 hours
- `0 9 * * 1` = Every Monday at 9 AM

**Pro tip:** Use [crontab.guru](https://crontab.guru) to build cron expressions!

---

#### 3. `permissions` - What the Workflow Can Do

```yaml
permissions:
  contents: write      # Can modify repository files
  pages: write         # Can deploy to GitHub Pages
  id-token: write      # Can request ID tokens (for deployments)
  pull-requests: write # Can comment on PRs
  issues: write        # Can create/modify issues
```

**Why needed?** Security - workflows run with limited permissions by default.

**Only request what you need!** Don't grant `write` access if you only need `read`.

---

#### 4. `jobs` - What to Do

A workflow can have multiple jobs that run in parallel or sequentially.

```yaml
jobs:
  build:                    # Job ID (unique identifier)
    name: Build App         # Human-readable name
    runs-on: ubuntu-latest  # Operating system

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build
        run: npm run build

  test:
    name: Run Tests
    runs-on: ubuntu-latest
    needs: build            # Wait for 'build' job to finish

    steps:
      - name: Run tests
        run: npm test
```

**Job properties:**
- `runs-on`: Which OS (ubuntu-latest, windows-latest, macos-latest)
- `needs`: Specify dependencies between jobs
- `if`: Conditional execution
- `strategy`: Run job multiple times with different configurations

---

#### 5. `steps` - Detailed Instructions

Each job contains steps - the actual commands to run.

**Two types of steps:**

**a) Using Actions (reusable components):**
```yaml
- uses: actions/checkout@v4          # Clones your repo
  with:
    fetch-depth: 0                   # Parameters for the action
```

**b) Running Commands:**
```yaml
- name: Install dependencies
  run: npm install                   # Any shell command

- name: Multiple commands
  run: |
    npm install
    npm run build
    echo "Build complete!"
```

**Step properties:**
- `name`: Description (shows in logs)
- `uses`: Use a pre-built action
- `run`: Run shell commands
- `with`: Parameters for actions
- `env`: Environment variables
- `if`: Conditional execution
- `id`: Give the step an ID for later reference

---

## Breaking Down Your Deploy Workflow

Let's analyze your deployment workflow section by section:

### Full Workflow Overview

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches:
      - main

permissions:
  contents: write
  pages: write
  id-token: write

jobs:
  build:
    # ... build job ...

  deploy:
    # ... deploy job ...
```

**What this does:** When you push to `main`, it builds your Docusaurus site and deploys it to GitHub Pages.

---

### Job 1: Build

```yaml
build:
  name: Build Docusaurus
  runs-on: ubuntu-latest
  defaults:
    run:
      working-directory: ./website
```

**Line by line:**
- `build:` - Job ID
- `name: Build Docusaurus` - What shows in GitHub UI
- `runs-on: ubuntu-latest` - Use Ubuntu Linux virtual machine
- `defaults.run.working-directory: ./website` - Run all commands in the `website` folder

**Why working-directory?** Your Docusaurus project is in a subfolder, not the root.

---

#### Step 1: Checkout Code

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0
```

**What it does:** Downloads your repository code to the virtual machine

**`fetch-depth: 0`:** Gets full git history (needed for some plugins that use git info)

**Without this step:** The VM would be empty - no code to work with!

---

#### Step 2: Setup Node.js

```yaml
- uses: actions/setup-node@v4
  with:
    node-version: 20
    cache: npm
    cache-dependency-path: './website/package-lock.json'
```

**What it does:** Installs Node.js version 20

**Why caching?**
- `cache: npm` - Saves `node_modules` between runs
- Speeds up subsequent runs (don't re-download packages)
- `cache-dependency-path` - Tells it where to find `package-lock.json`

**Speed boost:** First run ~2 minutes, cached runs ~30 seconds!

---

#### Step 3: Install Dependencies

```yaml
- name: Install dependencies
  run: npm ci
```

**What it does:** Installs packages from `package-lock.json`

**Why `npm ci` instead of `npm install`?**
- `npm ci` = Clean install (deletes node_modules first)
- Faster in CI environments
- Uses exact versions from lock file (reproducible builds)
- `npm install` = Can update versions, slower

**Use `npm ci` in CI/CD, `npm install` locally!**

---

#### Step 4: Build Website

```yaml
- name: Build website
  run: npm run build
```

**What it does:** Runs your build script (creates static HTML/CSS/JS files)

**This runs:** `docusaurus build` (from your package.json scripts)

**Output:** Creates `website/build/` folder with your static site

---

#### Step 5: Upload Build Artifact

```yaml
- name: Upload Build Artifact
  uses: actions/upload-pages-artifact@v3
  with:
    path: website/build
```

**What it does:** Saves the built website so the deploy job can use it

**Why needed?** Jobs run on separate virtual machines - need to pass data between them

**Artifact:** A zip file of your build folder stored temporarily by GitHub

---

### Job 2: Deploy

```yaml
deploy:
  name: Deploy to GitHub Pages
  needs: build
  permissions:
    pages: write
    id-token: write
  environment:
    name: github-pages
    url: ${{ steps.deployment.outputs.page_url }}
  runs-on: ubuntu-latest
```

**Line by line:**
- `needs: build` - Wait for build job to finish successfully
- `permissions` - Only needs deployment permissions
- `environment: github-pages` - Uses GitHub Pages environment
- `url: ${{ ... }}` - Shows deployment URL in GitHub UI

---

#### Deploy Step

```yaml
- name: Deploy to GitHub Pages
  id: deployment
  uses: actions/deploy-pages@v4
```

**What it does:** Takes the artifact from build job and deploys to GitHub Pages

**`id: deployment`:** Gives this step an ID so we can reference its outputs

**That's it!** The action handles everything:
- Takes the uploaded artifact
- Deploys to GitHub Pages
- Returns the URL

---

## Common Workflow Patterns

### Pattern 1: Run Tests on Pull Requests

```yaml
name: Run Tests

on:
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

      - name: Comment on PR
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '❌ Tests failed! Please fix before merging.'
            })
```

**What this does:** Runs tests on every PR and comments if tests fail.

---

### Pattern 2: Matrix Builds (Test Multiple Versions)

```yaml
name: Test Multiple Node Versions

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        node-version: [18, 20, 22]

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}

      - run: npm ci
      - run: npm test
```

**What this does:** Runs tests on 9 combinations (3 OS × 3 Node versions)

**Use case:** Ensure your code works across different environments.

---

### Pattern 3: Scheduled Tasks

```yaml
name: Sync Data Daily

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
  workflow_dispatch:      # Also allow manual trigger

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Sync from API
        env:
          API_KEY: ${{ secrets.API_KEY }}
        run: npm run sync

      - name: Commit changes
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add .
          git diff --quiet || git commit -m "chore: daily sync"
          git push
```

**What this does:** Runs daily to sync data, commits changes if any.

---

### Pattern 4: Conditional Execution

```yaml
name: Deploy

on:
  push:
    branches: [main, develop]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to Production
        if: github.ref == 'refs/heads/main'
        run: npm run deploy:prod

      - name: Deploy to Staging
        if: github.ref == 'refs/heads/develop'
        run: npm run deploy:staging
```

**What this does:** Different deployment based on which branch triggered the workflow.

**Conditions:**
- `if: success()` - Only if previous steps succeeded
- `if: failure()` - Only if previous steps failed
- `if: always()` - Always run
- `if: github.event_name == 'pull_request'` - Only on PRs

---

## Environment Variables and Secrets

### Using Environment Variables

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    env:
      NODE_ENV: production
      API_URL: https://api.example.com

    steps:
      - name: Build with env vars
        run: npm run build
```

### Using Secrets (Sensitive Data)

**Store in GitHub:** Repo Settings → Secrets and variables → Actions → New repository secret

**Use in workflow:**
```yaml
steps:
  - name: Deploy
    env:
      API_KEY: ${{ secrets.API_KEY }}
      DATABASE_URL: ${{ secrets.DATABASE_URL }}
    run: npm run deploy
```

**⚠️ Never commit secrets to your code!** Always use GitHub Secrets.

---

## Debugging Workflows

### View Logs

1. Go to repo → **Actions** tab
2. Click on a workflow run
3. Click on a job
4. Click on a step to see detailed logs

### Debug with Intermediate Steps

```yaml
- name: Debug - Print variables
  run: |
    echo "Node version: $(node --version)"
    echo "Working directory: $(pwd)"
    echo "Files: $(ls -la)"
    echo "Branch: ${{ github.ref }}"
```

### Enable Debug Logging

1. Repo Settings → Secrets → New secret
2. Name: `ACTIONS_STEP_DEBUG`
3. Value: `true`

This shows extra debug info in logs.

---

## Writing Your First Workflow

Let's write a simple workflow from scratch:

### Goal: Auto-format Code on Push

**Step 1:** Create `.github/workflows/format.yml`

**Step 2:** Define the trigger
```yaml
name: Auto Format Code

on:
  push:
    branches:
      - main
```

**Step 3:** Add a job
```yaml
jobs:
  format:
    runs-on: ubuntu-latest
```

**Step 4:** Add steps
```yaml
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install Prettier
        run: npm install -g prettier

      - name: Format code
        run: prettier --write "**/*.{js,jsx,ts,tsx,md}"

      - name: Commit changes
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add .
          git diff --quiet || git commit -m "style: auto-format code"
          git push
```

**Complete workflow:**
```yaml
name: Auto Format Code

on:
  push:
    branches:
      - main

jobs:
  format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - run: npm install -g prettier

      - name: Format code
        run: prettier --write "**/*.{js,jsx,ts,tsx,md}"

      - name: Commit changes
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add .
          git diff --quiet || git commit -m "style: auto-format code"
          git push
```

**What it does:** Every push to main formats all code files and commits the changes.

---

## Best Practices

### 1. Use Specific Action Versions

```yaml
# ❌ Bad - can break if action updates
- uses: actions/checkout@v4

# ✅ Good - pin to exact version
- uses: actions/checkout@v4.1.1

# ✅ Better - pin to commit SHA (most secure)
- uses: actions/checkout@8ade135a41bc03ea155e62e844d188df1ea18608
```

### 2. Don't Repeat Yourself

Use composite actions or reusable workflows:

```yaml
# .github/workflows/reusable-test.yml
on:
  workflow_call:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm test

# .github/workflows/main.yml
jobs:
  test:
    uses: ./.github/workflows/reusable-test.yml
```

### 3. Cache Dependencies

```yaml
- uses: actions/setup-node@v4
  with:
    cache: npm  # Always cache!
```

Saves time and GitHub Actions minutes!

### 4. Fail Fast

```yaml
strategy:
  fail-fast: true  # Stop all jobs if one fails
  matrix:
    node: [18, 20, 22]
```

### 5. Use Conditions to Save Resources

```yaml
- name: Deploy
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
  run: npm run deploy
```

Only run expensive operations when needed!

---

## Common Issues and Solutions

### Issue: "Permission denied" when pushing

**Solution:** Add `contents: write` permission:
```yaml
permissions:
  contents: write
```

### Issue: Workflow doesn't trigger

**Checklist:**
- ✅ File is in `.github/workflows/`
- ✅ File has `.yml` or `.yaml` extension
- ✅ YAML syntax is valid
- ✅ Trigger event matches what you're doing
- ✅ Branch names are correct

### Issue: Action not found

**Solution:** Check action exists and version is correct:
```yaml
# Check on GitHub Marketplace
- uses: actions/checkout@v4  # Verify version exists
```

### Issue: Slow workflows

**Solutions:**
- Cache dependencies (`cache: npm`)
- Use `npm ci` instead of `npm install`
- Run jobs in parallel (remove `needs:`)
- Use matrix strategically

---

## Your Deploy Workflow Summary

Let's review what your workflow does step-by-step:

```
1. Trigger: You push to main branch
2. GitHub: Starts an Ubuntu virtual machine
3. Build Job:
   - Clone your repo
   - Install Node.js 20
   - Run npm ci (install dependencies)
   - Run npm run build (build Docusaurus site)
   - Upload the build folder as an artifact
4. Deploy Job:
   - Wait for build to finish
   - Download the build artifact
   - Deploy to GitHub Pages
   - Show the URL in GitHub UI
5. Done! Your site is live at github.io
```

**Time:** Usually 2-3 minutes for first run, ~1 minute with caching.

---

## Next Steps

**Now you can:**
1. ✅ Understand any GitHub Actions workflow
2. ✅ Modify your deploy workflow if needed
3. ✅ Write your own workflows from scratch
4. ✅ Debug issues when they occur

**Practice projects:**
- Add a workflow to run tests on PRs
- Create a scheduled job to sync data
- Auto-label issues based on content
- Send notifications when deployments succeed

---

## Resources

**Official Documentation:**
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)

**Useful Actions:**
- [GitHub Marketplace](https://github.com/marketplace?type=actions)
- [actions/checkout](https://github.com/actions/checkout)
- [actions/setup-node](https://github.com/actions/setup-node)
- [actions/cache](https://github.com/actions/cache)

**Tools:**
- [actionlint](https://github.com/rhysd/actionlint) - Lint your workflows
- [act](https://github.com/nektos/act) - Run actions locally

**Learning:**
- [GitHub Actions Examples](https://github.com/sdras/awesome-actions)

---

## Appendix: YAML Quick Reference

### Basic Syntax

```yaml
# Comment

key: value                    # String
number: 42                    # Number
boolean: true                 # Boolean

list:                         # List/Array
  - item1
  - item2

object:                       # Object/Map
  key1: value1
  key2: value2

multiline: |                  # Multiline string (preserves newlines)
  Line 1
  Line 2

folded: >                     # Folded string (newlines become spaces)
  This is a very long
  string that wraps

array_short: [1, 2, 3]        # Inline array
object_short: {key: value}    # Inline object
```

### GitHub Actions Specific

```yaml
# Reference other values
${{ github.ref }}             # Branch name
${{ secrets.API_KEY }}        # Secret
${{ steps.step-id.outputs.value }}  # Step output

# Expressions
${{ github.ref == 'refs/heads/main' }}  # Condition
${{ matrix.version }}         # Matrix value
```

Remember: YAML is whitespace-sensitive! Use 2 spaces for indentation (not tabs).

---

You're now ready to automate your entire development workflow! 🚀