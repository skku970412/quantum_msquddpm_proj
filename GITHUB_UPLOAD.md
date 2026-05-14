# GitHub Upload Status

Target repository:

```text
git@github.com:skku970412/quantum_msquddpm_proj.git
```

Local status:

- Git repository initialized in this project folder.
- Branch: `main`
- Remote: `origin`
- Commit created: `Add quantum MSQuDDPM-lite benchmark`
- Benchmark results verified before commit.

Push command:

```bash
cd /home/work/llama_young/gpu-quddpm-lite-benchmark
git push -u origin main
```

Current blocker:

```text
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.
```

This means the server's SSH key is not registered with the GitHub account or repository. Add the server public key to GitHub, then run the push command again.

Print the server public key:

```bash
ssh-keygen -y -f /home/work/.ssh/id_rsa
```

GitHub path:

```text
GitHub -> Settings -> SSH and GPG keys -> New SSH key
```

After adding the key, verify and push:

```bash
cd /home/work/llama_young/gpu-quddpm-lite-benchmark
git ls-remote origin
git push -u origin main
```

