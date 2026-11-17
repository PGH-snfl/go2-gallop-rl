"""Check source files without importing training or simulation modules."""
import ast
import json
import subprocess
import sys
import tokenize

from gpd.paths import ROOT


def main():
    source_dirs = ("gpd", "isaac_task", "scripts", "tests")
    sources = sorted(path for folder in source_dirs for path in (ROOT / folder).rglob("*.py"))
    configs = sorted((ROOT / "configs").glob("*.json"))
    errors = []
    for path in sources:
        try:
            with tokenize.open(path) as stream:
                ast.parse(stream.read(), filename=str(path), feature_version=(3, 11))
        except (SyntaxError, UnicodeError, LookupError, ValueError, OSError) as error:
            errors.append(f"{path.relative_to(ROOT)}: {error}")
    for path in configs:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            errors.append(f"{path.relative_to(ROOT)}: {error}")
    if errors:
        print("\n".join(errors), flush=True)
        return 1
    print(f"Syntax OK: {len(sources)} Python files, {len(configs)} JSON configs.", flush=True)
    commands = [
        [sys.executable, "-B", "-m", "ruff", "check", "--no-cache", "--config",
         "configs/ruff-edit.toml", *source_dirs],
        ["bash", "-n", "isaac.sh"],
        ["bash", "-n", "gallop.sh"],
        ["bash", "-n", "setup_isaac.sh"],
        [sys.executable, "-B", "-m", "unittest", "tests.test_portable_handoff"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    print("Code checks passed. No training, simulation or artifact downloads executed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
