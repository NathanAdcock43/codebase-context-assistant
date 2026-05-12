from __future__ import annotations

"""
ROLE: Scan this project, extract architectural headers and code shape, and generate system map outputs.
LAYER: diagnostics
FLOW: map_generation

INPUTS:
- repository Python files
- architecture header docstrings
- import declarations
- class declarations
- function declarations
- system-map output paths

OUTPUTS:
- docs/system_map.md
- docs/system_map.json
- header and structure warnings for scanned modules

UPSTREAM:
- repository source files
- architectural header discipline
- developer local execution

DOWNSTREAM:
- architecture review
- repo cleanup and header correction
- future maintenance and re-entry into the codebase
- AI-assisted project context review

OWNS:
- source file scanning eligibility
- Python module header extraction
- architecture header parsing
- warning generation for header issues
- import inventory extraction
- internal import detection
- class and function inventory extraction
- markdown system-map rendering
- JSON system-map rendering

DOES_NOT_OWN:
- runtime assistant behavior
- source file mutation
- header rewriting
- architecture decisions
- vector indexing
- LLM retrieval
- FastAPI routing
- test execution

SIDE_EFFECTS:
- reads repository source files
- writes docs/system_map.md
- writes docs/system_map.json
- prints console summary output

STATE:
  reads:
    - repository source files
  writes:
    - docs/system_map.md
    - docs/system_map.json

NOTES:
- This script keeps the project architecture inspectable as the codebase evolves.
- It should remain documentation-focused and must not take on runtime assistant logic.
- Header placement rule: from __future__ import annotations comes first, then the architecture header.
"""

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_MD = ROOT / "docs" / "system_map.md"
OUTPUT_JSON = ROOT / "docs" / "system_map.json"

INCLUDE_DIRS = {
    "code_context",
    "tests",
}

INTERNAL_IMPORT_ROOTS = {
    "code_context",
}

EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "data",
    "docs",
    "logs",
    ".code_context_index",
    ".chroma",
}

HEADER_KEYS = {
    "ROLE",
    "LAYER",
    "FLOW",
    "INPUTS",
    "OUTPUTS",
    "UPSTREAM",
    "DOWNSTREAM",
    "OWNS",
    "DOES_NOT_OWN",
    "SIDE_EFFECTS",
    "STATE",
    "NOTES",
}

LAYER_ORDER = [
    "entry",
    "api",
    "agent",
    "retrieval",
    "indexing",
    "scanner",
    "parser",
    "chunking",
    "drift",
    "config",
    "core_model",
    "diagnostics",
    "tests",
    "shared",
    "unknown",
]


@dataclass(slots=True)
class ModuleMap:
    path: str
    role: str | None = None
    layer: str | None = None
    flow: str | None = None
    header_fields: dict[str, Any] = field(default_factory=dict)
    imports: list[str] = field(default_factory=list)
    internal_imports: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def should_scan(path: Path) -> bool:
    if path.suffix != ".py":
        return False

    if path.name == "__init__.py":
        return False

    relative_parts = set(path.relative_to(ROOT).parts)

    if relative_parts & EXCLUDE_DIRS:
        return False

    return any(part in INCLUDE_DIRS for part in relative_parts)


def parse_header(docstring: str | None) -> dict[str, Any]:
    """
    Parse the architectural header from a module docstring.

    Supports scalar values, bullet-list blocks, and simple nested map blocks.
    """
    if not docstring:
        return {}

    lines = [line.rstrip() for line in docstring.splitlines()]
    fields: dict[str, Any] = {}
    i = 0

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if not stripped:
            i += 1
            continue

        if ":" not in stripped:
            i += 1
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()

        if key not in HEADER_KEYS:
            i += 1
            continue

        value = value.strip()

        if value:
            fields[key] = value
            i += 1
            continue

        block_lines: list[str] = []
        i += 1

        while i < len(lines):
            next_raw = lines[i]
            next_stripped = next_raw.strip()

            if not next_stripped:
                block_lines.append(next_raw)
                i += 1
                continue

            if ":" in next_stripped:
                maybe_key = next_stripped.split(":", 1)[0].strip()
                indent = len(next_raw) - len(next_raw.lstrip())
                if maybe_key in HEADER_KEYS and indent == 0:
                    break

            block_lines.append(next_raw)
            i += 1

        fields[key] = _parse_header_block(block_lines)

    return fields


def _parse_header_block(lines: list[str]) -> Any:
    meaningful = [line for line in lines if line.strip()]
    if not meaningful:
        return []

    stripped = [line.strip() for line in meaningful]

    if all(line.startswith("- ") for line in stripped):
        return [line[2:].strip() for line in stripped]

    nested: dict[str, Any] = {}
    current_key: str | None = None

    for raw in meaningful:
        stripped_line = raw.strip()

        if stripped_line.startswith("- "):
            item = stripped_line[2:].strip()
            if current_key is None:
                nested.setdefault("_items", [])
                nested["_items"].append(item)
            else:
                existing = nested.get(current_key)
                if existing is None:
                    nested[current_key] = [item]
                elif isinstance(existing, list):
                    existing.append(item)
                else:
                    nested[current_key] = [existing, item]
            continue

        if ":" in stripped_line:
            subkey, subvalue = stripped_line.split(":", 1)
            subkey = subkey.strip()
            subvalue = subvalue.strip()
            current_key = subkey

            if subvalue:
                nested[subkey] = subvalue
            else:
                nested[subkey] = []
            continue

    if nested:
        return nested

    return "\n".join(stripped)


def get_import_name(node: ast.AST) -> list[str]:
    names: list[str] = []

    if isinstance(node, ast.Import):
        for alias in node.names:
            names.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level > 0:
            prefix = "." * node.level
            names.append(f"{prefix}{module}" if module else prefix)
        else:
            names.append(module)

    return names


def is_internal_import(name: str) -> bool:
    top = name.lstrip(".").split(".", 1)[0]
    return top in INTERNAL_IMPORT_ROOTS


def parse_python_file(path: Path) -> ModuleMap:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    rel_path = path.relative_to(ROOT).as_posix()

    warnings: list[str] = []

    header_docstring, header_first_statement = _extract_header_docstring_and_position(tree)
    header = parse_header(header_docstring)

    if not header_docstring:
        first_docstring = ast.get_docstring(tree)
        if first_docstring:
            warnings.append("missing_architecture_header")
        else:
            warnings.append("missing_module_docstring")
    elif not header_first_statement:
        warnings.append("header_docstring_not_first_statement")

    if header_docstring and not header:
        warnings.append("module_docstring_present_but_header_not_parsed")

    if header and "ROLE" not in header:
        warnings.append("missing_role")
    if header and "LAYER" not in header:
        warnings.append("missing_layer")

    imports: list[str] = []
    internal_imports: list[str] = []
    classes: list[str] = []
    functions: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in get_import_name(node):
                if name:
                    imports.append(name)
                    if is_internal_import(name):
                        internal_imports.append(name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append(node.name)

    return ModuleMap(
        path=rel_path,
        role=_coerce_str(header.get("ROLE")),
        layer=_coerce_str(header.get("LAYER")) or "unknown",
        flow=_coerce_str(header.get("FLOW")),
        header_fields=header,
        imports=sorted(set(imports)),
        internal_imports=sorted(set(internal_imports)),
        classes=classes,
        functions=functions,
        warnings=warnings,
    )


def _extract_header_docstring_and_position(tree: ast.Module) -> tuple[str | None, bool]:
    """
    Return the first architectural-looking module-level docstring and whether
    it is the first non-__future__ statement in the module.
    """
    if not tree.body:
        return None, False

    first_meaningful_index = 0
    while first_meaningful_index < len(tree.body):
        node = tree.body[first_meaningful_index]
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            first_meaningful_index += 1
            continue
        break

    if first_meaningful_index >= len(tree.body):
        return None, False

    first_meaningful_node = tree.body[first_meaningful_index]
    first_is_docstring = _node_is_string_expr(first_meaningful_node)

    if first_is_docstring:
        value = _string_expr_value(first_meaningful_node)
        if value and _looks_like_arch_header(value):
            return value, True

    for node in tree.body:
        if _node_is_string_expr(node):
            value = _string_expr_value(node)
            if value and _looks_like_arch_header(value):
                return value, False

    return None, False


def _node_is_string_expr(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr):
        return False

    value = getattr(node, "value", None)

    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return True

    return isinstance(value, ast.Str)


def _string_expr_value(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Expr):
        return None

    value = getattr(node, "value", None)

    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value

    if isinstance(value, ast.Str):
        return value.s

    return None


def _looks_like_arch_header(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    return "ROLE:" in stripped and "LAYER:" in stripped


def collect_modules() -> list[ModuleMap]:
    modules: list[ModuleMap] = []

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue

        if should_scan(path):
            modules.append(parse_python_file(path))

    return modules


def layer_sort_key(module: ModuleMap) -> tuple[int, str, str]:
    layer = module.layer or "unknown"

    try:
        index = LAYER_ORDER.index(layer)
    except ValueError:
        index = len(LAYER_ORDER) - 1

    return index, layer, module.path


def render_markdown(modules: list[ModuleMap]) -> str:
    lines: list[str] = []
    lines.append("# System Map")
    lines.append("")
    lines.append("Generated by `src/code_context/scripts/build_system_map.py`.")
    lines.append("")

    grouped: dict[str, list[ModuleMap]] = {}
    for module in modules:
        key = module.layer or "unknown"
        grouped.setdefault(key, []).append(module)

    for layer in sorted(grouped.keys(), key=_layer_group_sort_key):
        lines.append(f"## Layer: {layer}")
        lines.append("")

        for module in sorted(grouped[layer], key=lambda item: item.path):
            lines.append(f"### `{module.path}`")
            lines.append("")
            lines.append(f"- Role: {module.role or 'Unspecified'}")
            lines.append(f"- Flow: {module.flow or 'Unspecified'}")

            if module.internal_imports:
                lines.append(f"- Internal imports: {', '.join(module.internal_imports)}")
            else:
                lines.append("- Internal imports: none")

            if module.classes:
                lines.append(f"- Classes: {', '.join(module.classes)}")
            else:
                lines.append("- Classes: none")

            if module.functions:
                lines.append(f"- Functions: {', '.join(module.functions)}")
            else:
                lines.append("- Functions: none")

            owns = module.header_fields.get("OWNS", [])
            does_not_own = module.header_fields.get("DOES_NOT_OWN", [])

            if isinstance(owns, list) and owns:
                lines.append("- Owns:")
                for item in owns:
                    lines.append(f"  - {item}")

            if isinstance(does_not_own, list) and does_not_own:
                lines.append("- Does not own:")
                for item in does_not_own:
                    lines.append(f"  - {item}")

            if module.warnings:
                lines.append("- Warnings:")
                for warning in module.warnings:
                    lines.append(f"  - {warning}")

            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _layer_group_sort_key(layer: str) -> tuple[int, str]:
    try:
        return LAYER_ORDER.index(layer), layer
    except ValueError:
        return len(LAYER_ORDER) - 1, layer


def write_outputs(modules: list[ModuleMap]) -> None:
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(modules, key=layer_sort_key)

    md = render_markdown(ordered)
    OUTPUT_MD.write_text(md, encoding="utf-8")

    json_payload = [asdict(module) for module in ordered]
    OUTPUT_JSON.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def print_summary(modules: list[ModuleMap]) -> None:
    warning_count = sum(1 for module in modules if module.warnings)

    print(f"Scanned {len(modules)} source files.")
    print(f"Wrote {OUTPUT_MD.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_JSON.relative_to(ROOT)}")

    if warning_count:
        print("")
        print("Modules with warnings:")
        for module in modules:
            if module.warnings:
                joined = ", ".join(module.warnings)
                print(f"  - {module.path}: {joined}")


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def main() -> None:
    modules = collect_modules()
    write_outputs(modules)
    print_summary(modules)


if __name__ == "__main__":
    main()