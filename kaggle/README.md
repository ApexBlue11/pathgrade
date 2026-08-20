# Kaggle kernels

Scripts that run the pipeline on Kaggle, kept in the repo so a run is
reproducible from source rather than from someone's notebook history.

| file | session | purpose |
|---|---|---|
| `preflight.py` | CPU | verifies the dataset mount, HF token, gated model access, openslide and the GDC query before any quota is spent |
| `extract_tpu.py` | TPU | streams all labelled slides from GDC, encodes with H-optimus-0, writes embeddings |

## Secrets do not survive an API push

H-optimus-0 is a gated HuggingFace repo, so extraction needs an `HF_TOKEN`.
Kaggle secrets are attached **per notebook through the UI**, and a kernel pushed
with `kaggle kernels push` does not inherit that attachment — the secrets
service is not provisioned for it at all, and `UserSecretsClient` fails with
`ConnectionError: Connection error trying to communicate with service`.

Each `push` also creates a new version, which can clear an attachment made
earlier. So the working order is:

1. `kaggle kernels push -p .` to create or update the notebook
2. open it on kaggle.com
3. **Add-ons > Secrets**, add `HF_TOKEN`, tick to attach it to this notebook
4. **Save & Run All** from the UI
5. monitor with `kaggle kernels status` / `kaggle kernels output`

Do not push again after step 3 without redoing it.

## Measured session facts

See `src/pathgrade/platform.py`. Briefly: TPU allows **1** concurrent session,
GPU **2**, CPU **5**; CPU sessions cannot run a 1B-parameter encoder;
`/kaggle/tmp` holds ~1.1 TB of scratch while `/kaggle/working` caps at ~20 GB
and is the only path that persists.
