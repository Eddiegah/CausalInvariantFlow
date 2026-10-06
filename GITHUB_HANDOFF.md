# GitHub publication handoff

The repository is initialized locally and committed at `ef681c5b682bae936988e0b6edb321c4f5ea3ae9`.

The Manus GitHub connector is not present on this device and the plain GitHub CLI has no authenticated session. Do not run authentication automatically. After enabling the GitHub connector or authenticating `gh` in the user-controlled environment, publish with:

```powershell
gh repo create CausalInvariantFlow --private --source . --remote origin --push
```

The requested repository name is `CausalInvariantFlow` and the intended visibility is private.
