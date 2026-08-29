# Third-party licences

Argus has **no licence of its own**. There is no `LICENSE` file, which under copyright law means
all rights are reserved — nobody may copy, modify or redistribute it.

That is deliberate, not an oversight. **The licence cannot be chosen until the Ultralytics
question below is resolved**, because one dependency constrains what Argus itself may be
licensed as.

> This document is engineering notes, not legal advice. Before shipping commercially, or before
> putting a `LICENSE` file in this repository, get the Ultralytics position reviewed by someone
> qualified.

---

## The blocking issue: Ultralytics YOLO is AGPL-3.0

`ultralytics` (and the bundled `yolo11n.pt` weights) is licensed **AGPL-3.0**, with a paid
Enterprise Licence as the alternative.

Per [ultralytics.com/license](https://www.ultralytics.com/license), checked 2026-08-08, using
Ultralytics YOLO code, models, architectures, training pipelines, **or trained/fine-tuned
models** requires either:

1. open-sourcing **the entire project** under AGPL-3.0, or
2. an Ultralytics Enterprise Licence.

Their own list of cases requiring an Enterprise Licence includes:

- internal business tools or private company applications
- any commercial product or service
- **SaaS platforms, APIs, or cloud systems that use YOLO behind the scenes**
- **embedded deployments in hardware, edge devices, robotics, cameras, or appliances**
- proprietary or closed-source software

Argus is an HTTP API using YOLO behind the scenes, intended for deployment on a Raspberry Pi at a
commercial weighbridge. On Ultralytics' stated reading that is squarely in the Enterprise column.

### What this rules out

**Argus cannot be MIT or Apache-2.0 licensed while `ultralytics` is a dependency.** A permissive
licence on a work incorporating AGPL-3.0 code is self-contradictory. Adding one would be worse
than having no licence, because it would assert something untrue.

### Things that do *not* avoid it

| Idea | Why it doesn't work |
|---|---|
| Export the model to ONNX and drop the `ultralytics` runtime | Ultralytics state that trained models are AGPL-3.0 in their own right. The export is a derivative of an AGPL model. |
| Train our own weights from scratch | Their FAQ addresses this directly — the obligation attaches to the architecture and training pipeline, not only the published weights. |
| Use it only internally | "Internal business tools or private company applications" is on their Enterprise list. |

Genuinely escaping AGPL means a **different detector**, not a different packaging of this one.

### Options

1. **License Argus AGPL-3.0.** Matches what the dependency already requires. Anyone deploying a
   derivative must publish source. Check any client contract before choosing this.
2. **Buy an Ultralytics Enterprise Licence.** Frees Argus to be licensed however you like,
   including proprietary. Costs money; needs arranging.
3. **Replace the detector** with a permissively-licensed one. Large change, and it would need its
   own issue and its own accuracy re-baseline.

**Current status: unresolved.** The repository is public and unlicensed until a decision is made.

---

## Other dependencies

Everything else is permissive and imposes no reciprocal obligation on Argus.

| Package | Licence | Notes |
|---|---|---|
| `ultralytics` | **AGPL-3.0** | ⚠️ see above |
| `paddleocr`, `paddlepaddle` | Apache-2.0 | |
| `fastapi`, `pydantic`, `loguru`, `tabulate` | MIT | |
| `uvicorn`, `httpx`, `numpy` | BSD-3-Clause | |
| `torch` | BSD-3-Clause | pulled in transitively by `ultralytics` |
| `opencv-python` | Apache-2.0 | |
| `pillow` | MIT-CMU (HPND) | |
| `requests`, `python-multipart` | Apache-2.0 | |
| `onnxruntime` | MIT | declared but currently unused |

> [!WARNING]
> **This table is indicative and was written from knowledge, not generated from the lockfile.**
> Before any external release, produce the authoritative list from the installed environment:
>
> ```bash
> uvx pip-licenses --format=markdown --with-urls --order=license
> ```
>
> Reconcile any difference and update this file. Do not cite this table to a customer as-is.

---

## Model weights

`yolo11n.pt` is committed to the repository. Two separate concerns:

- **Licensing** — the weights are AGPL-3.0, as above.
- **Security** — `.pt` files are pickles, and `torch.load` executes arbitrary code on load.
  Committing the file rather than fetching it at runtime contains the risk, but there is no
  checksum verification. Tracked in the audit as P2.7.

---

## Data protection

Not a software licence, but it belongs in the same conversation.

Licence plate + timestamp + location is **personal data** under India's DPDP Act 2023, and under
GDPR where an EU nexus exists. Argus currently sends images to Plate Recognizer and NVIDIA — a
cross-border transfer to third-party processors — with no Data Processing Agreement, no retention
policy and no purpose notice.

This must be resolved before any live commercial weighing, independently of the code licence. See
`ISSUE_AUDIT_SUMMARY.md` (P2.6).
