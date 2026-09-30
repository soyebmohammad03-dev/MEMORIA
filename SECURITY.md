# Security policy

MEMORIA is a local research library. It has no server, no authentication and no network
listener. The only network access is `memoria.neural.fetch_model`, which the user calls
explicitly to download pinned model files from Hugging Face; the files are verified by size and
SHA-256 and a corrupt download is discarded.

## Supported versions

Only the latest commit on `main` receives fixes.

## Reporting a vulnerability

Please do not put exploit details in a public issue. Use the repository's **Security** tab
("Report a vulnerability") when it is enabled. If it is not, open an issue that says only that you
have a security report, and the maintainer will arrange a private channel. Include the affected
module, a minimal reproduction and the impact you see once you have one. This is a
single-maintainer research project, so there is no formal response-time guarantee.

## Scope

In scope: unsafe file or process behaviour (path handling in the artifact store, archive or model
file handling), integrity failures that let a tampered artifact, log or model file pass
verification, and dependency issues with a demonstrable effect on MEMORIA.

Out of scope: results that are scientifically wrong but not a security defect (please open a
regular issue), and attacks that require write access to the user's own artifact directory.

## Data

MEMORIA studies generated worlds. It is not designed to store personal data, and no part of it
should be pointed at real users' memories without your own privacy review.
