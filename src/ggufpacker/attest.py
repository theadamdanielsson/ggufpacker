"""Derivation attestations: prove a GGUF quant derives bit-exactly from a
source model via a recorded recipe, and emit/verify an in-toto Statement
saying so.

The attestation is an unsigned in-toto Statement v1 (JSON) with a custom
predicate. The core assertion is machine-checkable: a verifier re-runs
`recipe` against `baseModel` with a llama-quantize build and byte-compares
the output's sha256 against `subject.digest.sha256`. `attest` never emits a
claim it has not just proven locally; `verify` never reports success without
re-deriving. See docs/derivation-attestation.md for the predicate spec.

Trust model: unlike SLSA provenance ("trust the builder"), this predicate is
verifiable by ANYONE with the pinned quantize build — re-derive and compare.
Cross-machine verification requires a deterministic quantize build
(ggml-org/llama.cpp#25353); with default builds, verification is expected to
succeed only on a binary matching the attester's.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .blobs import sha256_file
from .layout import GGUFParseError, parse_layout
from .quantizer import Quantizer
from .recipe import (
    Recipe,
    detect_overrides,
    guess_recipe,
    override_cli_name,
    tensor_type_map,
)

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = (
    "https://github.com/theadamdanielsson/ggufpacker/attestation/gguf-derivation/v0"
)
DETERMINISM_EVIDENCE = "https://github.com/ggml-org/llama.cpp/pull/25353"


class AttestError(RuntimeError):
    """No recipe reproduced the quant bit-exactly; nothing was attested."""


class AttestationInvalid(ValueError):
    """The attestation file is not a statement this version can verify."""


class VerifyFailed(RuntimeError):
    """Re-derivation did not reproduce the attested bytes."""


@dataclass
class Attested:
    statement: dict[str, Any]
    recipe: Recipe
    seconds: float


def _prove_recipe(
    quant_path: Path,
    source_path: Path,
    imatrix_path: Path | None,
    quantizer: Quantizer,
    explicit: Recipe | None,
    workdir: Path,
    log,
) -> tuple[Recipe, str, float]:
    """Find and prove a recipe whose output sha256-matches the quant.

    Returns (recipe, quant_sha256, seconds spent quantizing). Raises
    AttestError when nothing reproduces — an attestation is never emitted on
    faith. Mirrors the packer's prove loop, minus the blob store."""
    quant_sha = sha256_file(quant_path)
    try:
        published_layout = parse_layout(quant_path)
    except GGUFParseError as e:
        raise AttestError(f"{quant_path.name}: not a parseable GGUF ({e})") from e

    if explicit is not None:
        candidates: list[Recipe] = [explicit]
    else:
        guess = guess_recipe(quant_path.name, published_layout)
        if guess is None:
            raise AttestError(
                f"{quant_path.name}: no recipe candidates (unknown type suffix "
                f"and unhelpful tensor histogram); pass --qtype"
            )
        use_imx = imatrix_path is not None
        candidates = [Recipe(qtype=c, use_imatrix=use_imx) for c in guess.candidates]

    published_map = tensor_type_map(published_layout)
    spent = 0.0
    # Portability convention: the imatrix is always passed as a BARE filename
    # with cwd at its directory. llama-quantize embeds the passed string in
    # the output header, so any other spelling produces machine-local bytes.
    # Artifacts originally quantized with a different imatrix path string
    # cannot match under this convention — attest refuses them (see the hint
    # in the refusal message) rather than emit a non-portable attestation.
    imx_arg = imatrix_path.name if imatrix_path else None
    imx_cwd = imatrix_path.parent if imatrix_path else None
    for recipe in candidates:
        out = workdir / f"attest-{recipe.qtype}.gguf"
        res = quantizer.run(
            source_path, out, recipe.qtype,
            imatrix=imx_arg if recipe.use_imatrix else None,
            cwd=imx_cwd if recipe.use_imatrix else None,
            token_embedding_type=recipe.token_embedding_type,
            output_tensor_type=recipe.output_tensor_type,
        )
        spent += res.seconds
        if res.returncode != 0:
            log(f"  {recipe.qtype}: llama-quantize failed (rc {res.returncode})")
            continue
        if sha256_file(out) == quant_sha:
            return recipe, quant_sha, spent

        # Base type produced different bytes: check whether the difference is
        # a CLI-overridable tensor-type mix (bartowski _L/_XL style) and retry.
        if recipe.token_embedding_type or recipe.output_tensor_type:
            log(f"  {recipe.qtype} (+overrides): bytes differ")
            continue
        try:
            regen_map = tensor_type_map(parse_layout(out))
        except GGUFParseError:
            continue
        overrides = detect_overrides(published_map, regen_map)
        if not overrides:
            log(f"  {recipe.qtype}: bytes differ, no overridable-tensor explanation")
            continue
        retry = Recipe(
            qtype=recipe.qtype,
            token_embedding_type=(
                override_cli_name(overrides["token_embd.weight"])
                if "token_embd.weight" in overrides else None
            ),
            output_tensor_type=(
                override_cli_name(overrides["output.weight"])
                if "output.weight" in overrides else None
            ),
            use_imatrix=recipe.use_imatrix,
        )
        res = quantizer.run(
            source_path, out, retry.qtype,
            imatrix=imx_arg if retry.use_imatrix else None,
            cwd=imx_cwd if retry.use_imatrix else None,
            token_embedding_type=retry.token_embedding_type,
            output_tensor_type=retry.output_tensor_type,
        )
        spent += res.seconds
        if res.returncode == 0 and sha256_file(out) == quant_sha:
            return retry, quant_sha, spent
        log(f"  {retry.qtype} (+detected overrides): bytes differ")

    hint = ""
    if imatrix_path is not None:
        embedded = _embedded_imatrix_path(quant_path)
        if embedded is not None and embedded != imatrix_path.name:
            hint = (
                f" (the file's header embeds imatrix path {embedded!r}; only "
                f"artifacts quantized with the bare cwd-relative filename "
                f"{imatrix_path.name!r} are portable and attestable — "
                f"re-quantize with `cd <dir> && llama-quantize --imatrix "
                f"{imatrix_path.name} ...`)"
            )
    raise AttestError(
        f"{quant_path.name}: no candidate recipe reproduced the file bit-exactly "
        f"with this llama-quantize build; refusing to attest{hint}"
    )


def _embedded_imatrix_path(quant_path: Path) -> str | None:
    """Best-effort read of the quantize.imatrix.file KV a quant's header
    embeds — the usual reason an otherwise-correct recipe fails to byte-match."""
    try:
        from gguf import GGUFReader

        reader = GGUFReader(quant_path)
        field = reader.fields.get("quantize.imatrix.file")
        if field is None:
            return None
        try:
            return str(field.contents())
        except AttributeError:  # older gguf-py without Field.contents()
            return bytes(field.parts[field.data[0]]).decode("utf-8", "replace")
    except Exception:
        return None


def _recipe_predicate(recipe: Recipe, imatrix_name: str | None) -> dict[str, Any]:
    cmd = ["llama-quantize"]
    if recipe.use_imatrix and imatrix_name:
        cmd += ["--imatrix", imatrix_name]
    if recipe.token_embedding_type:
        cmd += ["--token-embedding-type", recipe.token_embedding_type]
    if recipe.output_tensor_type:
        cmd += ["--output-tensor-type", recipe.output_tensor_type]
    cmd += ["<baseModel>", "<output>", recipe.qtype]
    d: dict[str, Any] = {
        "quantType": recipe.qtype,
        "outputFormat": "gguf",
        "useImatrix": recipe.use_imatrix,
        "command": " ".join(cmd),
    }
    if recipe.token_embedding_type:
        d["tokenEmbeddingType"] = recipe.token_embedding_type
    if recipe.output_tensor_type:
        d["outputTensorType"] = recipe.output_tensor_type
    return d


def attest(
    quant_path: str | Path,
    source_path: str | Path,
    imatrix_path: str | Path | None = None,
    llama_quantize: str | None = None,
    qtype: str | None = None,
    token_embedding_type: str | None = None,
    output_tensor_type: str | None = None,
    llama_cpp_ref: str | None = None,
    deterministic_build: bool = False,
    source_uri: str | None = None,
    source_download_url: str | None = None,
    workdir: str | Path | None = None,
    log=print,
) -> Attested:
    """Prove-then-emit: re-derive the quant from the source, byte-compare,
    and only on a match build the in-toto Statement."""
    import tempfile

    # Resolved to absolute up front: the quantize subprocess runs with cwd at
    # the imatrix's directory (portability convention), so relative arguments
    # would silently point elsewhere.
    quant_path = Path(quant_path).resolve()
    source_path = Path(source_path).resolve()
    imatrix_path = Path(imatrix_path).resolve() if imatrix_path else None
    for p in (quant_path, source_path, *( [imatrix_path] if imatrix_path else [] )):
        if not p.is_file():
            raise FileNotFoundError(f"no such file: {p}")

    quantizer = Quantizer.locate(llama_quantize)
    explicit = None
    if qtype:
        explicit = Recipe(
            qtype=qtype,
            token_embedding_type=token_embedding_type,
            output_tensor_type=output_tensor_type,
            use_imatrix=imatrix_path is not None,
        )
    elif token_embedding_type or output_tensor_type:
        raise AttestError("--token-embedding-type/--output-tensor-type require --qtype")

    with tempfile.TemporaryDirectory(dir=quant_path.parent) as td:
        recipe, quant_sha, seconds = _prove_recipe(
            quant_path, source_path, imatrix_path, quantizer, explicit,
            Path(td), log,
        )

    # Canonical identity: `uri` (purl form, e.g. pkg:huggingface/org/model@rev)
    # plus a concrete downloadLocation lets a verifier anchor the base model to
    # its published identity instead of trusting the filename. Without them the
    # statement proves derivation from these bytes, not from a canonical model.
    base_model: dict[str, Any] = {
        "name": source_path.name,
        "digest": {"sha256": sha256_file(source_path)},
        "sizeBytes": source_path.stat().st_size,
    }
    if source_uri:
        base_model["uri"] = source_uri
    if source_download_url:
        base_model["downloadLocation"] = source_download_url
    predicate: dict[str, Any] = {
        "baseModel": base_model,
        "recipe": _recipe_predicate(recipe, imatrix_path.name if imatrix_path else None),
        "builder": {
            "tool": "ggml-org/llama.cpp/llama-quantize",
            "buildIdentity": {
                "binarySha256": quantizer.sha256,
                "versionBanner": quantizer.version,
                **({"gitRef": llama_cpp_ref} if llama_cpp_ref else {}),
            },
        },
        "reproducibility": {
            # True only when the attester asserts the binary was built with
            # the deterministic quantize flag; cross-machine verification is
            # expected to succeed only then.
            "deterministic": deterministic_build,
            "determinismEvidence": DETERMINISM_EVIDENCE,
            "reDerivedDigest": {"sha256": quant_sha},
        },
        "attestedBy": f"ggufpacker {__version__}",
        "producedAt": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if imatrix_path:
        predicate["recipe"]["imatrix"] = {
            "name": imatrix_path.name,
            "digest": {"sha256": sha256_file(imatrix_path)},
            "sizeBytes": imatrix_path.stat().st_size,
        }

    statement = {
        "_type": STATEMENT_TYPE,
        "subject": [
            {"name": quant_path.name, "digest": {"sha256": quant_sha}},
        ],
        "predicateType": PREDICATE_TYPE,
        "predicate": predicate,
    }
    return Attested(statement=statement, recipe=recipe, seconds=seconds)


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
# argv safety for attested strings that reach the llama-quantize command line:
# type names are alnum/underscore only — no dashes (flag injection), no
# separators, no whitespace.
_TYPE_ARG = re.compile(r"^[A-Za-z0-9_]{1,32}$")


def _check_name(path_label: str, name: Any) -> str:
    """Attested file names are always plain sibling filenames. Anything that
    could traverse (`..`, separators, absolute paths) is rejected before it
    reaches a filesystem join — verify must never read outside search_dir."""
    if (not isinstance(name, str) or not name or name in (".", "..")
            or "/" in name or "\\" in name or name != name.strip()
            or Path(name).is_absolute()):
        raise AttestationInvalid(f"{path_label}: unsafe file name {name!r}")
    return name


def _check_digest(path_label: str, ref: Any) -> str:
    sha = ref.get("digest", {}).get("sha256") if isinstance(ref, dict) else None
    if not isinstance(sha, str) or not _HEX64.match(sha):
        raise AttestationInvalid(f"{path_label}: missing or malformed sha256 digest")
    return sha


def load_statement(path: str | Path) -> dict[str, Any]:
    """Parse and strictly validate a statement. Every field that later
    reaches the filesystem or the quantize argv is validated HERE, so the
    verifier can treat the returned structure as safe. Any statement this
    version does not fully understand is a refusal, never a guess."""
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
    subject_sha = _check_digest(f"{path}: subject", subs[0])

    pred = data.get("predicate")
    if not isinstance(pred, dict):
        raise AttestationInvalid(f"{path}: predicate is not an object")
    for key in ("baseModel", "recipe", "builder", "reproducibility"):
        if not isinstance(pred.get(key), dict):
            raise AttestationInvalid(f"{path}: predicate missing {key!r}")

    _check_name(f"{path}: baseModel", pred["baseModel"].get("name"))
    _check_digest(f"{path}: baseModel", pred["baseModel"])

    recipe = pred["recipe"]
    qtype = recipe.get("quantType")
    if not isinstance(qtype, str) or not _TYPE_ARG.match(qtype):
        raise AttestationInvalid(f"{path}: malformed quantType {qtype!r}")
    for k in ("tokenEmbeddingType", "outputTensorType"):
        v = recipe.get(k)
        if v is not None and (not isinstance(v, str) or not _TYPE_ARG.match(v)):
            raise AttestationInvalid(f"{path}: malformed {k} {v!r}")
    if recipe.get("useImatrix"):
        imx = recipe.get("imatrix")
        if not isinstance(imx, dict):
            raise AttestationInvalid(
                f"{path}: recipe uses an imatrix but attests none"
            )
        _check_name(f"{path}: imatrix", imx.get("name"))
        _check_digest(f"{path}: imatrix", imx)

    # The attester's core assertion is subject == reDerivedDigest; a statement
    # where they disagree is self-contradictory and refused outright.
    rd = pred["reproducibility"].get("reDerivedDigest")
    if rd is not None:
        rd_sha = rd.get("sha256") if isinstance(rd, dict) else None
        if not isinstance(rd_sha, str) or not _HEX64.match(rd_sha):
            raise AttestationInvalid(f"{path}: malformed reDerivedDigest")
        if rd_sha != subject_sha:
            raise AttestationInvalid(
                f"{path}: reDerivedDigest does not equal the subject digest — "
                f"the statement contradicts itself"
            )
    return data


def verify(
    statement_path: str | Path,
    llama_quantize: str | None = None,
    search_dir: str | Path | None = None,
    timeout: float | None = 3600.0,
    check_source: bool = False,
    log=print,
) -> None:
    """Re-derive and byte-compare. Raises VerifyFailed/FileNotFoundError/
    AttestationInvalid on any failure; returns None on proven success.

    Files (base model, imatrix, and optionally the quant itself) are located
    by their attested names in search_dir (default: the attestation's
    directory) and their digests are checked before the quantize runs.
    `timeout` bounds the quantize subprocess (statements are untrusted input;
    a re-derivation should not run unbounded). `check_source` additionally
    resolves the attested baseModel downloadLocation/uri and requires the
    published digest to equal the attested one — without it, the statement
    proves derivation from the attested bytes, not from a canonical model."""
    import tempfile

    statement_path = Path(statement_path).resolve()
    data = load_statement(statement_path)
    base = (Path(search_dir) if search_dir else statement_path.parent).resolve()

    pred = data["predicate"]
    subject = data["subject"][0]
    want_sha = subject["digest"]["sha256"]

    if check_source:
        _check_source_identity(pred["baseModel"], log)

    src = base / pred["baseModel"]["name"]
    if not src.is_file():
        raise FileNotFoundError(f"base model not found: {src}")
    got = sha256_file(src)
    if got != pred["baseModel"]["digest"]["sha256"]:
        raise VerifyFailed(
            f"{src.name}: base model sha256 mismatch "
            f"(attested {pred['baseModel']['digest']['sha256'][:16]}..., "
            f"found {got[:16]}...)"
        )

    imatrix = None
    if pred["recipe"].get("useImatrix"):
        imx_ref = pred["recipe"].get("imatrix")
        if not imx_ref:
            raise AttestationInvalid(
                f"{statement_path}: recipe uses an imatrix but attests none"
            )
        imatrix = base / imx_ref["name"]
        if not imatrix.is_file():
            raise FileNotFoundError(f"imatrix not found: {imatrix}")
        got = sha256_file(imatrix)
        if got != imx_ref["digest"]["sha256"]:
            raise VerifyFailed(
                f"{imatrix.name}: imatrix sha256 mismatch "
                f"(attested {imx_ref['digest']['sha256'][:16]}..., found {got[:16]}...)"
            )

    # If the attested quant file itself is present, check it up front: a
    # tampered artifact should be reported as such, not as a derivation gap.
    quant_on_disk = base / subject["name"]
    if quant_on_disk.is_file():
        got = sha256_file(quant_on_disk)
        if got != want_sha:
            raise VerifyFailed(
                f"{subject['name']}: file on disk does not match the attested "
                f"sha256 (attested {want_sha[:16]}..., found {got[:16]}...)"
            )

    quantizer = Quantizer.locate(llama_quantize)
    attested_bin = pred["builder"]["buildIdentity"].get("binarySha256", "")
    if attested_bin and attested_bin != quantizer.sha256:
        note = (
            "attested build is deterministic; a matching deterministic build "
            "should still reproduce it"
            if pred["reproducibility"].get("deterministic")
            else "attested build is NOT marked deterministic; reproduction with "
            "a different binary is not expected to succeed"
        )
        log(f"note: llama-quantize differs from the attesting binary ({note})")

    r = pred["recipe"]
    with tempfile.TemporaryDirectory(dir=base) as td:
        out = Path(td) / "rederived.gguf"
        # Same portability convention as attest: imatrix passed as its bare
        # attested name, cwd at its directory, so the embedded header string
        # matches the attester's byte-for-byte.
        res = quantizer.run(
            src, out, r["quantType"],
            imatrix=imatrix.name if imatrix else None,
            cwd=imatrix.parent if imatrix else None,
            token_embedding_type=r.get("tokenEmbeddingType"),
            output_tensor_type=r.get("outputTensorType"),
            timeout=timeout,
        )
        if res.returncode != 0:
            raise VerifyFailed(
                f"llama-quantize failed (rc {res.returncode}): "
                f"{res.output_tail[-300:]}"
            )
        got = sha256_file(out)
    if got != want_sha:
        # A mismatch with the ATTESTER'S OWN binary is a different finding
        # than a mismatch with some other build: same deterministic function,
        # digest-checked inputs, different output — the attested file cannot
        # be the output of the attested recipe. Report them distinctly.
        if attested_bin and attested_bin == quantizer.sha256:
            raise VerifyFailed(
                f"TAMPER-EVIDENT: the attesting binary itself "
                f"(sha256 {attested_bin[:16]}...) re-derives "
                f"{got[:16]}..., not the attested {want_sha[:16]}... — "
                f"the attested file is not the output of the attested recipe"
            )
        raise VerifyFailed(
            f"INCONCLUSIVE mismatch: re-derivation gives {got[:16]}..., "
            f"attested {want_sha[:16]}... — this verifier's llama-quantize "
            f"differs from the attesting binary "
            f"(attested sha256 {attested_bin[:16] or 'unrecorded'}...); "
            + ("the attester claimed a deterministic build, so a deterministic "
               "build of the attested gitRef should settle it"
               if pred["reproducibility"].get("deterministic")
               else "the attester did not claim a deterministic build, so a "
               "mismatch on a different binary is expected")
        )


_HF_RESOLVE = re.compile(
    r"^https://huggingface\.co/([\w.-]+/[\w.-]+)/resolve/([\w.-]+)/([^?]+)$"
)


def _check_source_identity(base_model: dict[str, Any], log=print) -> None:
    """--check-source: require the attested baseModel digest to equal the
    digest of the published file it names. v0 supports huggingface.co
    downloadLocation URLs, checked via the /raw/ endpoint: for LFS files that
    returns a small pointer carrying the sha256, so no model download."""
    loc = base_model.get("downloadLocation")
    if not isinstance(loc, str) or not loc:
        raise AttestationInvalid(
            "statement has no baseModel.downloadLocation to check against; "
            "without it the statement proves derivation from the attested "
            "bytes, not from a canonical published model"
        )
    m = _HF_RESOLVE.match(loc)
    if not m:
        raise AttestationInvalid(
            f"--check-source supports huggingface.co resolve URLs only in v0 "
            f"(got {loc!r})"
        )
    repo, rev, filename = m.groups()
    published = _hf_published_sha256(f"https://huggingface.co/{repo}/raw/{rev}/{filename}")
    attested = base_model["digest"]["sha256"]
    if published != attested:
        raise VerifyFailed(
            f"baseModel identity check failed: {repo}@{rev}/{filename} "
            f"publishes sha256 {published[:16]}..., statement attests "
            f"{attested[:16]}..."
        )
    log(f"source identity confirmed: baseModel digest matches "
        f"{repo}@{rev}/{filename} on huggingface.co")


def _hf_published_sha256(raw_url: str) -> str:
    """sha256 of a Hugging Face file via its /raw/ endpoint: LFS files return
    a pointer containing the oid; small non-LFS files return their content,
    which is hashed directly. Separated for testability."""
    import hashlib
    import urllib.request

    with urllib.request.urlopen(raw_url, timeout=60) as resp:  # noqa: S310
        head = resp.read(512)
        if head.startswith(b"version https://git-lfs"):
            m = re.search(rb"oid sha256:([0-9a-f]{64})", head)
            if not m:
                raise VerifyFailed(f"{raw_url}: LFS pointer without an oid")
            return m.group(1).decode()
        h = hashlib.sha256(head)
        while chunk := resp.read(1 << 20):
            h.update(chunk)
        return h.hexdigest()
