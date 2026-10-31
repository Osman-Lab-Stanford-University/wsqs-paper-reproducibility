# Data privacy and safe use

This repository contains **no observed household records, respondent identifiers, household-level predictions or raw interview files**. The 25 records under `example/` are synthetically generated demonstration examples only. The `results/aggregate/` folder contains aggregated statistics rather than household-level exports.

The empirical study's data are restricted. Researchers with authorized access must hold their input data and execution outputs in a separate secure directory, *outside this Git repository*. Do not commit completed surveys, household feature matrices, sensitive pseudonymous records, private execution logs, fitted models that could leak personal information or credentials.

The `.gitignore` is an accident-prevention aid, **not** a substitute for reviewing files before uploading files to GitHub. A file committed once remains in Git history even if later removed. Inspect repository files before publishing. No external researcher is granted data access or new consent rights by this repository.

The public demo is not drawn from the original study and must not be used to claim replication of its published predictive performance.
