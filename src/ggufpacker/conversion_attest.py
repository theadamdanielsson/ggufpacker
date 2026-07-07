"""Conversion attestations: prove an F16 GGUF derives bit-exactly from a
Hugging Face safetensors snapshot via convert_hf_to_gguf.py, and emit/verify an
in-toto Statement saying so. The sibling of gguf-derivation/v0, one edge up the
chain: chained, they attest the whole path from the published safetensors (the
Hugging Face trust root) to the deployed quant.

Same discipline as the quant path: `attest_conversion` never emits a claim it
has not just proven locally (re-run the conversion, byte-compare); `verify_
conversion` never reports success without re-deriving. The difference is the
source: a multi-file snapshot CLOSURE, not one F16. Every file in the closure is
digest-pinned, and the snapshot directory name is itself an input (metadata
heuristics parse it), recorded as `directoryName`.

See docs/conversion-attestation.md for the predicate spec.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .attest import (
    AttestationInvalid,
    AttestError,
    VerifyFailed,
    _check_digest,
    _check_name,
    _check_source_identity,
)
from .attest import (
    load_statement as load_quant_statement,
)
from .attest import (
    verify as verify_quant,
)
from .blobs import sha256_file
from .converter import Converter

# --model-name values recovered from an attested command reach the converter
# argv; only plain names are allowed there, never anything option-shaped.
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][\w.+-]*$")

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = (
    "https://github.com/theadamdanielsson/ggufpacker/attestation/gguf-conversion/v0"
)
CONVERSION_EVIDENCE = (
    "https://github.com/theadamdanielsson/gguf-quant-determinism"
    "/blob/main/.github/workflows/conversion.yml"
)


@dataclass
class ConversionAttested:
    statement: dict[str, Any]
    seconds: float


def _closure_files(source_dir: Path) -> list[dict[str, Any]]:
    """The input closure: every non-hidden file in the snapshot, name-relative,
    sorted, each digest-pinned. README.md is included when present (the
    converter reads the model card into KV metadata)."""
    files = []
    for p in sorted(source_dir.rglob("*")):
        rel_parts = p.relative_to(source_dir).parts
        if p.is_file() and not any(part.startswith(".") for part in rel_parts):
            files.append({
                "name": "/".join(rel_parts),
                "digest": {"sha256": sha256_file(p)},
                "sizeBytes": p.stat().st_size,
            })
    if not files:
        raise AttestError(f"{source_dir}: empty snapshot (no files to attest)")
    return files


def attest_conversion(
    f16_path: str | Path,
    source_dir: str | Path,
    llama_cpp_dir: str | None = None,
    converter: str | None = None,
    python: str | None = None,
    llama_cpp_ref: str | None = None,
    deterministic: bool = True,
    source_uri: str | None = None,
    source_revision: str | None = None,
    model_name: str | None = None,
    workdir: str | Path | None = None,
    timeout: float | None = 3600.0,
    log=print,
) -> ConversionAttested:
    """Prove-then-emit: re-run convert_hf_to_gguf on the snapshot, byte-compare
    to `f16_path`, and only on a match build the Statement."""
    import tempfile

    f16_path = Path(f16_path).resolve()
    source_dir = Path(source_dir).resolve()
    if not f16_path.is_file():
        raise FileNotFoundError(f"no such file: {f16_path}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"no such snapshot directory: {source_dir}")

    conv = Converter.locate(llama_cpp_dir, converter, python, llama_cpp_ref)
    want_sha = sha256_file(f16_path)
    closure = _closure_files(source_dir)

    wd = Path(workdir).resolve() if workdir else f16_path.parent
    with tempfile.TemporaryDirectory(dir=wd) as td:
        out = Path(td) / "rederived-f16.gguf"
        res = conv.run(source_dir, out, "f16", model_name=model_name, timeout=timeout)
        if res.returncode != 0:
            raise AttestError(
                f"convert_hf_to_gguf failed (rc {res.returncode}); refusing to "
                f"attest:\n{res.output_tail[-400:]}"
            )
        got = sha256_file(out)
    if got != want_sha:
        raise AttestError(
            f"{f16_path.name}: conversion of {source_dir.name} re-derives "
            f"{got[:16]}..., not the file's {want_sha[:16]}... — refusing to "
            f"attest. The committed F16 is not the output of this converter on "
            f"this snapshot (check the snapshot directory name and that "
            f"README.md matches; both feed the header)."
        )

    source: dict[str, Any] = {
        "directoryName": source_dir.name,
        "files": closure,
    }
    if source_uri:
        source["uri"] = source_uri
    if source_revision:
        source["revision"] = source_revision

    predicate: dict[str, Any] = {
        "source": source,
        "recipe": {
            "transform": "convert_hf_to_gguf",
            "outputType": "f16",
            "command": (
                "python convert_hf_to_gguf.py --outtype f16 "
                + (f"--model-name {model_name} " if model_name else "")
                + "--outfile <output> <sourceDir>"
            ),
            "conventions": [
                "the snapshot directory is named exactly the repo basename; "
                "general.name/basename/finetune/size_label derive from it",
                "README.md, when present, is part of the input closure "
                "(read into KV metadata)",
                "tensor order follows input iteration order; a permuted shard "
                "index permutes the output",
            ],
        },
        "builder": {
            "tool": "ggml-org/llama.cpp/convert_hf_to_gguf.py",
            "buildIdentity": {
                "converterSha256": conv.sha256,
                **({"gitRef": conv.git_ref} if conv.git_ref else {}),
                **({"python": conv.python_version} if conv.python_version else {}),
                **({"keyLibraries": conv.key_libraries} if conv.key_libraries else {}),
            },
        },
        "reproducibility": {
            "deterministic": deterministic,
            "determinismEvidence": CONVERSION_EVIDENCE,
            "reDerivedDigest": {"sha256": want_sha},
        },
        "attestedBy": f"ggufpacker {__version__}",
        "producedAt": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    statement = {
        "_type": STATEMENT_TYPE,
        "subject": [{"name": f16_path.name, "digest": {"sha256": want_sha}}],
        "predicateType": PREDICATE_TYPE,
        "predicate": predicate,
    }
    return ConversionAttested(statement=statement, seconds=res.seconds)


def load_conversion_statement(path: str | Path) -> dict[str, Any]:
    """Parse and strictly validate a conversion statement. Every field that
    later reaches the filesystem is validated here; anything this version does
    not fully understand is a refusal, never a guess."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such attestation: {path}")
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise AttestationInvalid(f"{path}: not valid JSON ({e})") from e
    if not isinstance(data, dict) or data.get("_type") != STATEMENT_TYPE:
        raise AttestationInvalid(f"{path}: not an in-toto Statement v1")
    if data.get("predicateType") != PREDICATE_TYPE:
        raise AttestationInvalid(
            f"{path}: unknown predicateType {data.get('predicateType')!r} "
            f"(this ggufpacker verifies {PREDICATE_TYPE!r})"
        )
    subs = data.get("subject")
    if not isinstance(subs, list) or len(subs) != 1 or not isinstance(subs[0], dict):
        raise AttestationInvalid(f"{path}: expected exactly one subject")
    _check_name(str(path), subs[0].get("name"))
    _check_digest(f"{path}: subject", subs[0])

    pred = data.get("predicate")
    if not isinstance(pred, dict):
        raise AttestationInvalid(f"{path}: predicate is not an object")
    for key in ("source", "recipe", "builder", "reproducibility"):
        if not isinstance(pred.get(key), dict):
            raise AttestationInvalid(f"{path}: predicate missing {key!r}")
    if not isinstance(pred["builder"].get("buildIdentity"), dict):
        raise AttestationInvalid(f"{path}: builder missing buildIdentity")

    command = pred["recipe"].get("command")
    if not isinstance(command, str):
        raise AttestationInvalid(f"{path}: recipe.command missing or not a string")
    model_name = _model_name_arg(command)
    if model_name is not None and not _MODEL_NAME.match(model_name):
        raise AttestationInvalid(
            f"{path}: unsafe --model-name in recorded command: {model_name!r}"
        )

    # The attester's core assertion is subject == reDerivedDigest; a statement
    # where they disagree is self-contradictory and refused outright.
    subject_sha = subs[0]["digest"]["sha256"]
    rd = pred["reproducibility"].get("reDerivedDigest")
    if rd is not None:
        rd_sha = rd.get("sha256") if isinstance(rd, dict) else None
        if not isinstance(rd_sha, str) or rd_sha != subject_sha:
            raise AttestationInvalid(
                f"{path}: reDerivedDigest does not equal the subject digest — "
                f"the statement contradicts itself"
            )

    src = pred["source"]
    dirname = src.get("directoryName")
    _check_name(f"{path}: source.directoryName", dirname)
    files = src.get("files")
    if not isinstance(files, list) or not files:
        raise AttestationInvalid(f"{path}: source.files must be a non-empty list")
    seen = set()
    for i, f in enumerate(files):
        if not isinstance(f, dict):
            raise AttestationInvalid(f"{path}: source.files[{i}] is not an object")
        # Closure names are relative POSIX paths within the snapshot dir; reject
        # anything that could traverse out of it or is absolute/backref.
        name = f.get("name")
        if (not isinstance(name, str) or not name or name != name.strip()
                or name.startswith("/") or ".." in Path(name).parts
                or Path(name).is_absolute()):
            raise AttestationInvalid(f"{path}: source.files[{i}] unsafe name {name!r}")
        if name in seen:
            raise AttestationInvalid(f"{path}: source.files duplicate {name!r}")
        seen.add(name)
        _check_digest(f"{path}: source.files[{i}]", f)
    return data


def verify_conversion(
    statement_path: str | Path,
    llama_cpp_dir: str | None = None,
    converter: str | None = None,
    python: str | None = None,
    search_dir: str | Path | None = None,
    timeout: float | None = 3600.0,
    check_source: bool = False,
    log=print,
) -> None:
    """Re-derive and byte-compare. Raises VerifyFailed/FileNotFoundError/
    AttestationInvalid on any failure; returns None on proven success.

    The snapshot files are located by their attested relative names under
    `search_dir/directoryName` (default search_dir: the attestation's
    directory) and every digest is checked before the conversion runs. The
    directory MUST carry the attested name -- it is a conversion input."""
    import tempfile

    statement_path = Path(statement_path).resolve()
    data = load_conversion_statement(statement_path)
    base = (Path(search_dir) if search_dir else statement_path.parent).resolve()

    pred = data["predicate"]
    subject = data["subject"][0]
    want_sha = subject["digest"]["sha256"]
    src = pred["source"]

    if check_source:
        _check_source_identity(_source_identity_view(src), log)

    snapshot = base / src["directoryName"]
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"snapshot directory not found: {snapshot} (the attested "
            f"directoryName is a conversion input and must be present verbatim)"
        )
    for f in src["files"]:
        fp = snapshot / f["name"]
        if not fp.is_file():
            raise FileNotFoundError(f"closure file not found: {fp}")
        got = sha256_file(fp)
        if got != f["digest"]["sha256"]:
            raise VerifyFailed(
                f"{f['name']}: closure sha256 mismatch (attested "
                f"{f['digest']['sha256'][:16]}..., found {got[:16]}...)"
            )

    # The closure is the whole recipe input: a file the attester never saw can
    # change the output (README.md does), so its presence means whatever runs
    # here is not the attested conversion. Refuse rather than mismatch later.
    attested_names = {f["name"] for f in src["files"]}
    for p in sorted(snapshot.rglob("*")):
        rel = p.relative_to(snapshot)
        if (p.is_file() and not any(part.startswith(".") for part in rel.parts)
                and rel.as_posix() not in attested_names):
            raise VerifyFailed(
                f"{rel.as_posix()}: file present in the snapshot but outside "
                f"the attested closure — the closure is the conversion input; "
                f"remove it or this is not the attested conversion"
            )

    # If the attested F16 is present, tampering with it is reported as tamper,
    # not as a derivation gap.
    f16_on_disk = base / subject["name"]
    if f16_on_disk.is_file():
        got = sha256_file(f16_on_disk)
        if got != want_sha:
            raise VerifyFailed(
                f"{subject['name']}: file on disk does not match the attested "
                f"sha256 (attested {want_sha[:16]}..., found {got[:16]}...)"
            )

    conv = Converter.locate(llama_cpp_dir, converter, python)
    attested_conv = pred["builder"]["buildIdentity"].get("converterSha256", "")
    if attested_conv and attested_conv != conv.sha256:
        log("note: convert_hf_to_gguf.py differs from the attesting converter "
            "(different llama.cpp commit); reproduction is expected only with a "
            "matching converter build")

    model_name = _model_name_arg(pred["recipe"].get("command", ""))
    with tempfile.TemporaryDirectory(dir=base) as td:
        out = Path(td) / "rederived-f16.gguf"
        res = conv.run(snapshot, out, "f16", model_name=model_name, timeout=timeout)
        if res.returncode != 0:
            raise VerifyFailed(
                f"convert_hf_to_gguf failed (rc {res.returncode}): "
                f"{res.output_tail[-300:]}"
            )
        got = sha256_file(out)
    if got != want_sha:
        if attested_conv and attested_conv == conv.sha256:
            raise VerifyFailed(
                f"TAMPER-EVIDENT: the attesting converter itself re-derives "
                f"{got[:16]}..., not the attested {want_sha[:16]}... — the "
                f"attested F16 is not the output of the attested conversion"
            )
        raise VerifyFailed(
            f"INCONCLUSIVE mismatch: re-derivation gives {got[:16]}..., attested "
            f"{want_sha[:16]}... — this verifier's convert_hf_to_gguf differs "
            f"from the attesting one (attested converterSha256 "
            f"{attested_conv[:16] or 'unrecorded'}...); build the attested "
            f"gitRef to settle it"
        )


def _source_identity_view(src: dict[str, Any]) -> dict[str, Any]:
    """--check-source anchors the whole snapshot to a published revision. v0
    checks the primary weights file (model.safetensors or the first .safetensors
    in the closure) against its Hugging Face published digest, reusing the quant
    path's HF resolver. A uri (purl) + revision must be present to build the URL."""
    uri = src.get("uri")
    revision = src.get("revision")
    weights = next((f for f in src["files"]
                    if f["name"].endswith(".safetensors")), None)
    if not weights:
        raise AttestationInvalid(
            "source closure has no .safetensors file to anchor identity against"
        )
    if not (isinstance(uri, str) and uri.startswith("pkg:huggingface/") and revision):
        raise AttestationInvalid(
            "--check-source needs source.uri (pkg:huggingface/org/model) and "
            "source.revision to locate the published file"
        )
    repo = uri[len("pkg:huggingface/"):].split("@", 1)[0]
    return {
        "digest": weights["digest"],
        "downloadLocation": (
            f"https://huggingface.co/{repo}/resolve/{revision}/{weights['name']}"
        ),
    }


def _model_name_arg(command: str) -> str | None:
    """Recover the --model-name value from the recorded command, if the attester
    used one (it changes general.name, so it is part of the recipe)."""
    parts = command.split()
    if "--model-name" in parts:
        i = parts.index("--model-name")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def verify_chain(
    quant_statement_path: str | Path,
    conversion_statement_path: str | Path,
    llama_quantize: str | None = None,
    llama_cpp_dir: str | None = None,
    converter: str | None = None,
    python: str | None = None,
    search_dir: str | Path | None = None,
    timeout: float | None = 3600.0,
    check_source: bool = False,
    log=print,
) -> None:
    """The full path from the Hugging Face snapshot to the deployed quant:
    require the two statements to be LINKED (the quant's attested baseModel
    digest IS the conversion's subject digest — same bytes, not same name),
    then re-derive both edges. Content links the chain; the linkage check
    runs on the parsed statements alone, before any tool is located or any
    subprocess runs.

    `check_source` anchors the root: the conversion source snapshot is checked
    against its published Hugging Face revision. The quant statement's own
    baseModel anchor is NOT checked here — inside a chain, the F16's
    provenance is the conversion attestation itself."""
    quant_data = load_quant_statement(quant_statement_path)
    conv_data = load_conversion_statement(conversion_statement_path)

    base_model = quant_data["predicate"]["baseModel"]
    f16_subject = conv_data["subject"][0]
    if base_model["digest"]["sha256"] != f16_subject["digest"]["sha256"]:
        raise VerifyFailed(
            f"chain broken: the quant statement attests baseModel sha256 "
            f"{base_model['digest']['sha256'][:16]}..., but the conversion "
            f"statement's subject is {f16_subject['digest']['sha256'][:16]}... "
            f"— these statements are not about the same F16"
        )
    if base_model["name"] != f16_subject["name"]:
        log(f"note: linked by digest, but named differently across statements "
            f"({base_model['name']!r} vs {f16_subject['name']!r})")
    log(f"link ok: quant baseModel == conversion subject "
        f"({f16_subject['digest']['sha256'][:16]}...)")

    verify_conversion(
        conversion_statement_path,
        llama_cpp_dir=llama_cpp_dir,
        converter=converter,
        python=python,
        search_dir=search_dir,
        timeout=timeout,
        check_source=check_source,
        log=log,
    )
    log("edge proven: snapshot -> F16 (conversion re-derived bit-exact)")

    verify_quant(
        quant_statement_path,
        llama_quantize=llama_quantize,
        search_dir=search_dir,
        timeout=timeout,
        check_source=False,
        log=log,
    )
    log("edge proven: F16 -> quant (quantization re-derived bit-exact)")
