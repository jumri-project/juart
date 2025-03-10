# JAIL

TBD

## Onboarding

### Install Required Software

To set up your development environment, install the following:

- **[VS Code](https://code.visualstudio.com/)** – Code editor with DevContainer support.
- **[Podman Desktop](https://podman-desktop.io/)** – GUI for managing Podman containers.

Additionally, install the **Dev Containers** extension in VS Code:

1. Open VS Code.
2. Go to **Extensions** (`Ctrl+Shift+X`).
3. Search for **Dev Containers**.
4. Install the extension from **Microsoft** or [visit the repo](https://github.com/devcontainers).

### Clone the Repository

Ensure you have Git installed, then clone the repository:

```
git clone git@github.com:inm-4/jail.git
cd jail
```

### Sharing Git Credentials with the Container

To authenticate with Git inside the container, install **Git** and **SSH** on your host machine:

```
git config --global user.name "Your Name"
git config --global user.email "your.email@address"
```

Ensure your SSH agent is running and your key is added:

```
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
```

For persistent SSH authentication, consider adding this to your `~/.bashrc` or `~/.zshrc`:

```
if [ -z "$SSH_AUTH_SOCK" ]; then
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_rsa
fi
```

More details: [Sharing Git Credentials with Containers](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials).

### Open the Project in a DevContainer

1. Open **VS Code**.
2. Press `Ctrl+Shift+P` and select **Dev Containers: Open Folder in Container…**.
3. Choose the cloned **QRAGE** project folder.
4. Wait for the container to build and start.

Once started, you're ready to code inside the containerized development environment.
