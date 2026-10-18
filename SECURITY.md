# Security

Do not commit Kaggle/API credentials, environment files, personal meal photos, health profiles, or private paths. `.gitignore` prevents common accidental additions but does not remove files already tracked by Git. Review staged changes before every push. If a credential is exposed, revoke or rotate it before addressing Git history.

Install dependencies from trusted package sources, keep them updated, and load only trusted model files. Restrict data ingestion to trusted local images during research. This repository does not provide a public upload endpoint.

For a sensitive vulnerability, use GitHub's private vulnerability reporting when enabled or the repository owner's published contact route. Do not post secrets or private images in public issues.
