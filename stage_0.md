# CHEAT SHEET

- **Author:** Pascal Bouda - CEA Research Engineer ([pascal.bouda@cea.fr](mailto:pascal.bouda@cea.fr))
- **Goal:** create a tiny Python project and push it to a new GitHub private repository.

## 0. Prerequisites

| | Windows | macOS | Linux (Debian/Ubuntu) |
|---|---|---|---|
| **Terminal to use** | **Git Bash** (installed with Git) | Terminal | Terminal |
| **Install Git** | https://git-scm.com (keep default options) | `git --version` (accepts the install prompt) | `sudo apt install git` |
| **Install Python** | https://python.org (tick **"Add Python to PATH"**) | https://python.org or `brew install python` | `sudo apt install python3` |
| **Run Python with** | `python` (or `py`) | `python3` | `python3` |

> 👉 **Windows users: run every command below in Git Bash** (Start menu → "Git Bash"), not in cmd or PowerShell. Then all commands are identical on the 3 systems.
> In this sheet, write `python3` instead of `python` on macOS/Linux.

Check the installs: `git --version` and `python --version` (or `python3 --version`).

Tell Git who you are (once per computer):

```bash
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"
```

## 1. Set up your SSH key (once per computer)

```bash
ssh-keygen -t ed25519 -C "your-github-email@example.com"
```

Press **Enter** to accept the default location (a passphrase is optional). Then display your **public** key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the whole line (starts with `ssh-ed25519`). On GitHub: **Settings → SSH and GPG keys → New SSH key**, give it a title (e.g. "School laptop"), paste, save.

Test the connection (type `yes` if asked to trust the host):

```bash
ssh -T git@github.com
```

Expected: `Hi YOUR-USERNAME! You've successfully authenticated…`

> ⚠️ Never share the **private** key (`id_ed25519`, no `.pub`). Only the `.pub` goes to GitHub.
> 💡 Handy copy command: macOS `pbcopy < ~/.ssh/id_ed25519.pub` · Windows (Git Bash) `clip < ~/.ssh/id_ed25519.pub` · Linux `xclip -sel clip < ~/.ssh/id_ed25519.pub`

## 2. Create the project

```bash
mkdir hello-python
cd hello-python
```

Create `hello.py` (with any text editor, e.g. VS Code) containing:

```python
print("Hello, world!")
```

Test it: `python hello.py` (macOS/Linux: `python3 hello.py`).

## 3. Create the repository on GitHub

1. **+ → New repository**, name: `hello-python`.
2. Leave it **empty**: do NOT tick "Add a README", ".gitignore" or "license".
3. Click **Create repository**, choose the **SSH** tab and copy the URL.

## 4. Initialize Git, commit, push

```bash
git init
git branch -M main
git add hello.py
git commit -m "First commit: hello world"
git remote add origin git@github.com:YOUR-USERNAME/hello-python.git
git push -u origin main
```

Refresh the GitHub page: your file is there. 🎉

## 5. Daily workflow (after the first push)

```bash
git status                      # what changed?
git add .                       # stage all changes
git commit -m "Describe change" # save a snapshot
git push                        # send to GitHub
```

## Quick vocabulary

| Word | Meaning |
|------|---------|
| **repository (repo)** | A project folder tracked by Git |
| **commit** | A saved snapshot with a message |
| **staging (`add`)** | Choosing which changes go in the next commit |
| **remote / origin** | The copy of the repo on GitHub |
| **push** | Send local commits to GitHub |
| **SSH key** | Key pair: private (stays on your PC) + public (given to GitHub) |