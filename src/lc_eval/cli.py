"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .dataset import prepare_dataset
from .generate import generate_with_vllm
from .judge import judge_generations
from .languages import canonical_language
from .postprocess import PROCESSORS, postprocess_generations
from .score import score_file


def _languages(values: Sequence[str]) -> list[str]:
    result = []
    for value in values:
        for item in value.split(","):
            language = canonical_language(item.strip())
            if language not in result:
                result.append(language)
    return result


def _images(values: Sequence[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("images must use LANGUAGE=IMAGE")
        language, image = value.split("=", 1)
        result[canonical_language(language.strip())] = image.strip()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lc-eval",
        description="Standalone vLLM generation and sandboxed multilingual LeetCode evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="normalize and translate dataset records")
    source = prepare.add_mutually_exclusive_group()
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--dataset", default="newfacade/LeetCodeDataset")
    prepare.add_argument("--split", default="test")
    prepare.add_argument("--languages", nargs="+", default=["python", "cpp", "c", "mojo"])
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path)
    prepare.add_argument("--max-tasks", type=int)
    prepare.add_argument("--max-tests", type=int)
    prepare.add_argument("--common-only", action=argparse.BooleanOptionalAction, default=True)

    generate = subparsers.add_parser("generate", help="generate candidates with offline vLLM")
    generate.add_argument("--tasks", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--run-config", type=Path)
    generate.add_argument("--model", required=True)
    generate.add_argument("--languages", nargs="+")
    generation = generate.add_argument_group("sampling")
    generation.add_argument("--n", type=int, default=1)
    generation.add_argument("--request-chunk-size", type=int, default=256)
    generation.add_argument("--max-tokens", type=int, default=4096)
    generation.add_argument("--temperature", type=float, default=0.0)
    generation.add_argument("--top-p", type=float, default=1.0)
    generation.add_argument("--top-k", type=int, default=-1)
    generation.add_argument("--min-p", type=float, default=0.0)
    generation.add_argument("--repetition-penalty", type=float, default=1.0)
    generation.add_argument("--seed", type=int, default=13)
    engine = generate.add_argument_group("vLLM engine")
    engine.add_argument("--tokenizer")
    engine.add_argument("--revision")
    engine.add_argument("--tokenizer-revision")
    engine.add_argument("--dtype", default="auto")
    engine.add_argument("--quantization")
    engine.add_argument("--tensor-parallel-size", type=int, default=1)
    engine.add_argument("--pipeline-parallel-size", type=int, default=1)
    engine.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    engine.add_argument("--max-model-len", type=int)
    engine.add_argument("--max-num-seqs", type=int)
    engine.add_argument("--cpu-offload-gb", type=float, default=0.0)
    engine.add_argument("--trust-remote-code", action="store_true")
    engine.add_argument("--enforce-eager", action="store_true")
    engine.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    generate.add_argument("--chat-template", type=Path)
    generate.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    postprocess = subparsers.add_parser(
        "postprocess", help="extract judge-ready code from raw model responses"
    )
    postprocess.add_argument("--generations", type=Path, required=True)
    postprocess.add_argument("--output", type=Path, required=True)
    postprocess.add_argument("--manifest", type=Path)
    postprocess.add_argument("--processor", choices=PROCESSORS, default="auto")
    postprocess.add_argument("--overwrite", action="store_true")

    judge = subparsers.add_parser("judge", help="compile and execute candidates in sandboxes")
    judge.add_argument("--tasks", type=Path, required=True)
    judge.add_argument("--generations", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--run-config", type=Path)
    judge.add_argument("--backend", choices=("docker", "podman", "apptainer", "local"), default="docker")
    judge.add_argument("--image", action="append", default=[], metavar="LANGUAGE=IMAGE")
    judge.add_argument("--workers", type=int, default=4)
    judge.add_argument("--compile-timeout", type=float, default=30.0)
    judge.add_argument("--run-timeout", type=float, default=5.0)
    judge.add_argument("--memory-mb", type=int, default=4096)
    judge.add_argument("--cpus", type=float, default=1.0)
    judge.add_argument("--pids-limit", type=int, default=64)
    judge.add_argument("--output-limit-bytes", type=int, default=65536)
    judge.add_argument("--allow-unsafe-local", action="store_true")
    judge.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    score = subparsers.add_parser("score", help="compute pass@k and failure summaries")
    score.add_argument("--judgments", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--k", nargs="+", type=int, default=[1])
    return parser


def _validate_positive(parser: argparse.ArgumentParser, **values: float | int) -> None:
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        parser.error(f"must be positive: {', '.join(invalid)}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        if args.max_tasks is not None and args.max_tasks <= 0:
            parser.error("--max-tasks must be positive")
        if args.max_tests is not None and args.max_tests <= 0:
            parser.error("--max-tests must be positive")
        languages = _languages(args.languages)
        result = prepare_dataset(
            input_path=args.input_jsonl,
            dataset_name=args.dataset,
            split=args.split,
            languages=languages,
            output_path=args.output,
            manifest_path=args.manifest or args.output.with_suffix(".manifest.json"),
            max_tasks=args.max_tasks,
            max_tests=args.max_tests,
            common_only=args.common_only,
        )
    elif args.command == "generate":
        _validate_positive(
            parser,
            n=args.n,
            request_chunk_size=args.request_chunk_size,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
        )
        if args.n > 1 and args.temperature <= 0:
            parser.error("--n > 1 requires --temperature > 0")
        result = generate_with_vllm(
            tasks_path=args.tasks,
            output_path=args.output,
            run_config_path=args.run_config or args.output.with_suffix(".config.json"),
            model=args.model,
            languages=_languages(args.languages) if args.languages else None,
            n=args.n,
            request_chunk_size=args.request_chunk_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            tokenizer=args.tokenizer,
            revision=args.revision,
            tokenizer_revision=args.tokenizer_revision,
            dtype=args.dtype,
            quantization=args.quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            cpu_offload_gb=args.cpu_offload_gb,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=args.enable_prefix_caching,
            chat_template=args.chat_template,
            resume=args.resume,
        )
    elif args.command == "postprocess":
        result = postprocess_generations(
            generations_path=args.generations,
            output_path=args.output,
            manifest_path=args.manifest or args.output.with_suffix(".manifest.json"),
            processor=args.processor,
            overwrite=args.overwrite,
        )
    elif args.command == "judge":
        _validate_positive(
            parser,
            workers=args.workers,
            compile_timeout=args.compile_timeout,
            run_timeout=args.run_timeout,
            memory_mb=args.memory_mb,
            cpus=args.cpus,
            pids_limit=args.pids_limit,
            output_limit_bytes=args.output_limit_bytes,
        )
        try:
            images = _images(args.image)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        result = judge_generations(
            tasks_path=args.tasks,
            generations_path=args.generations,
            output_path=args.output,
            config_path=args.run_config or args.output.with_suffix(".config.json"),
            backend=args.backend,
            images=images,
            workers=args.workers,
            compile_timeout=args.compile_timeout,
            run_timeout=args.run_timeout,
            memory_mb=args.memory_mb,
            cpus=args.cpus,
            pids_limit=args.pids_limit,
            output_limit_bytes=args.output_limit_bytes,
            allow_unsafe_local=args.allow_unsafe_local,
            resume=args.resume,
        )
    elif args.command == "score":
        if any(k <= 0 for k in args.k):
            parser.error("all --k values must be positive")
        result = score_file(args.judgments, args.output, sorted(set(args.k)))
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
