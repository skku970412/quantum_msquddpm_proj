# GitHub Upload Status

Target repository:

```text
https://github.com/skku970412/quantum_msquddpm_proj.git
```

Local status:

- Git repository initialized in this project folder.
- Branch: `main`
- Remote: `origin` set to HTTPS
- Commit created: `Add quantum MSQuDDPM-lite benchmark`
- Benchmark results verified before commit.

Push command:

```bash
cd /home/work/llama_young/gpu-quddpm-lite-benchmark
git push -u origin main
```

Current blocker:

```text
remote: No anonymous write access.
fatal: Authentication failed for 'https://github.com/skku970412/quantum_msquddpm_proj.git/'
```

This means HTTPS works, but the server does not have GitHub credentials. GitHub no longer accepts account passwords for Git over HTTPS, so use a Personal Access Token.

Recommended manual push:

```bash
cd /home/work/llama_young/gpu-quddpm-lite-benchmark
unset GIT_ASKPASS SSH_ASKPASS VSCODE_GIT_ASKPASS_NODE VSCODE_GIT_ASKPASS_EXTRA_ARGS VSCODE_GIT_ASKPASS_MAIN VSCODE_GIT_IPC_HANDLE
git push -u origin main
```

When prompted:

```text
Username: skku970412
Password: paste a GitHub Personal Access Token, not the account password
```

Token options:

- Fine-grained token: repository `skku970412/quantum_msquddpm_proj`, Contents: Read and write.
- Classic token: `repo` scope.

GitHub path:

```text
GitHub -> Settings -> Developer settings -> Personal access tokens
```

Check the remote:

```bash
cd /home/work/llama_young/gpu-quddpm-lite-benchmark
git remote -v
```
