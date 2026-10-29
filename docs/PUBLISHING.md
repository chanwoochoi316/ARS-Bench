# Publishing ARS-Bench on GitHub

These commands are instructions for the repository owner. Editing this guide does not create a GitHub repository or push any files.

## 1. Choose the repository contents

Publish the code, documentation, configuration examples, copied prompt resources, provenance information, tests, and the explicitly synthetic examples.

The supplied `.gitignore` excludes local datasets, models, indexes, experiment outputs, `.env` variants, interpreter paths, local setup notes, and machine-specific verification reports. The publishing archive follows the same exclusions. No project-wide license has been selected by this update; preserve upstream attribution when deciding how to distribute copied components.

## 2. Initialize a local repository

Run this from the ARS-Bench directory. Skip `git init` if this directory is already its own Git repository.

```bash
cd /path/to/ARS-Bench
git init -b main
```

If Git has no author identity configured, set it for this repository:

```bash
git config user.name 'YOUR NAME'
git config user.email 'YOUR GITHUB EMAIL OR NOREPLY ADDRESS'
```

Inspect the proposed contents:

```bash
git status --short --untracked-files=all
git status --short --ignored
```

Stage the intended public files explicitly:

```bash
git add .gitignore README.md pyproject.toml ars ars_bench configs docs scripts tests COPY_PROVENANCE.json
git add examples/synthetic_result.json examples/synthetic_report.json examples/synthetic_report.csv
git diff --cached --stat
git diff --cached
```

Check that the staged diff contains no actual API keys, local data, checkpoints, or experiment outputs. `.gitignore` does not remove files that were already tracked. To unstage an unwanted file before the initial commit while keeping its local copy:

```bash
git rm --cached -- PATH_TO_FILE
```

Commit the reviewed files:

```bash
git commit -m 'Initial ARS-Bench release'
```

## 3. Push to the existing ARS-Bench repository

The project repository is [chanwoochoi316/ARS-Bench](https://github.com/chanwoochoi316/ARS-Bench). After configuring your Git author identity and committing the reviewed files, connect it with:

```bash
git remote add origin https://github.com/chanwoochoi316/ARS-Bench.git
```

If `origin` is already configured, verify it with `git remote get-url origin`; use `git remote set-url origin URL` only when it needs updating. Authenticate using GitHub CLI (`gh auth login`) or a supported credential helper. GitHub account passwords are not accepted for HTTPS Git operations; use a personal access token when Git requests a password. Never put the token in a URL or commit it to a file.

Inspect the remote before uploading:

```bash
git ls-remote origin
```

If the repository is empty, upload the initial commit:

```bash
git push -u origin main
```

If the remote already has commits, fetch and review its history before integrating the local project. Do not force-push over existing work.

## 4. Alternative: create a repository with GitHub CLI

If GitHub CLI is installed, authenticate and create a repository from the local repository:

```bash
gh auth login
gh repo create YOUR_ACCOUNT/ARS-Bench --public --source=. --remote=origin --push
```

Replace `YOUR_ACCOUNT` with your GitHub username or organization. Use `--private` instead of `--public` for a private repository. This command creates the remote repository and uploads the committed files.

## 5. Alternative: create an empty repository on GitHub

If GitHub CLI is unavailable:

1. Create a repository named `ARS-Bench` on the GitHub website.
2. Choose its visibility.
3. Leave the initial README, `.gitignore`, and license options unchecked; the local project provides its files.
4. Add the remote URL and push from the server.

For SSH authentication:

```bash
git remote add origin git@github.com:YOUR_ACCOUNT/ARS-Bench.git
git push -u origin main
```

For HTTPS, use GitHub-supported credential authentication:

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/ARS-Bench.git
git push -u origin main
```

Do not include a token in the remote URL. If `origin` already exists, inspect `git remote -v` and use that remote, or update it with `git remote set-url origin URL`.

## 6. Publish later changes

Review and stage the specific updated files, then commit and push:

```bash
git status --short
git add README.md docs/PUBLISHING.md
git diff --cached
git commit -m 'Update documentation'
git push
```

## Large assets and fresh clones

The code repository excludes local `data/`, `models/`, `indexes/`, and `runs/` directories. A fresh clone must prepare those separately, following the README. GitHub blocks ordinary Git files larger than 100 MiB; use a separate distribution channel or Git LFS when deliberately distributing large assets.

## Official references

- [Add locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [GitHub CLI: gh repo create](https://cli.github.com/manual/gh_repo_create)
- [Add files and handle large files](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
- [Authenticate to GitHub](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github)
