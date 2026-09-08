"""
test_anti_template.py

Automated proof that the PATIENT reconstruction runtime (reconstruct_cli.py /
reconstruct_native_truedepth.py -> patient_native_fusion.py ->
patient_texture_baker.py -> render_back_validator.py) does not import, call,
or otherwise depend at runtime on any generic template/mannequin asset or
module. Run: ai-engine/venv/bin/python ai-engine/test_anti_template.py
"""
import ast
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

FORBIDDEN_MODULES = {
    "gnm_identity_fit",
    "gnm_face_shell",
    "gnm_correspondence",
    "gnm_vertex_trust",
    "gnm_dense_nose",
    "gnm_width_correction",
    "gnm_texture_atlas",
    "gnm_texture_bake",
    # Wrapper that itself imports the whole GNM chain above — caught once
    # already via the transitive check when this was reintroduced as a
    # "try GNM first, fall back to sparse" path; listed directly too so a
    # bare top-level `import reconstruct_gnm_fullhead` fails immediately
    # without relying on the transitive walk.
    "reconstruct_gnm_fullhead",
}
FORBIDDEN_SYMBOLS = {"run_gnm_reconstruction"}
FORBIDDEN_ASSET_SUBSTRINGS = (
    "gnm_head_v3.npz",
    "template_positions.npy",
    "template_vertex_positions.npy",
)

# The patient-facing runtime entry points only — legacy demo/test scripts
# elsewhere in ai-engine/ are explicitly allowed to keep using the GNM
# template (per the audit's own instruction: "Các file legacy có thể giữ
# lại nếu cần cho demo/test cũ, nhưng TUYỆT ĐỐI không được để chúng chạy
# trong reconstruction của bệnh nhân thật").
PATIENT_RUNTIME_FILES = [
    "reconstruct_cli.py",
    "reconstruct_native_truedepth.py",
    "patient_native_fusion.py",
    "patient_texture_baker.py",
    "render_back_validator.py",
]


def collect_imports(py_file: Path, module_level_only: bool = True) -> set[str]:
    """`module_level_only=True` (default) only counts imports that execute
    unconditionally the instant this file is imported — i.e. direct
    top-level statements, not ones nested inside a function/method body.
    A function-local `import` (e.g. `solve_camera_pose`'s own lazy
    `from gnm_correspondence import ...`) only runs if that SPECIFIC
    function is actually called — a file that merely defines such a
    function without calling it does not pull the dependency in at
    runtime, so counting it as a hard dependency would be a false
    positive. Set `module_level_only=False` for a stricter transitive scan
    when actually auditing whether a forbidden symbol is reachable at all,
    not just eagerly imported."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    modules = set()
    nodes = tree.body if module_level_only else list(ast.walk(tree))
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


def collect_string_literals(py_file: Path) -> list[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append(node.value)
    return literals


def main() -> int:
    failures: list[str] = []

    for filename in PATIENT_RUNTIME_FILES:
        py_file = CURRENT_DIR / filename
        if not py_file.exists():
            failures.append(f"MISSING expected patient-runtime file: {filename}")
            continue

        # D-strictgate — these 5 files ARE the whole gate: unlike the
        # transitive walk below (which stays module_level_only=True to
        # avoid flagging an incidental helper's unrelated lazy import, e.g.
        # detect_pose.py's own unrelated use of gnm_correspondence), a
        # forbidden import ANYWHERE in one of these exact files — lazy,
        # function-local, wrapped in a try/except, however hidden — must
        # fail loudly. This is deliberately stricter than the transitive
        # check because reconstruct_cli.py has now twice had a lazy,
        # function-local `from reconstruct_gnm_fullhead import ...` land in
        # its hot path specifically BECAUSE the module-level-only check
        # couldn't see it.
        imports = collect_imports(py_file, module_level_only=False)
        bad_imports = imports & FORBIDDEN_MODULES
        if bad_imports:
            failures.append(f"{filename}: imports forbidden generic-template module(s) (incl. lazy/function-local imports): {sorted(bad_imports)}")

        source = py_file.read_text(encoding="utf-8")
        for symbol in FORBIDDEN_SYMBOLS:
            if symbol in source:
                failures.append(f"{filename}: references forbidden symbol '{symbol}'")

        literals = collect_string_literals(py_file)
        for lit in literals:
            for bad_asset in FORBIDDEN_ASSET_SUBSTRINGS:
                if bad_asset in lit:
                    failures.append(f"{filename}: references forbidden generic-template asset path containing '{bad_asset}' (literal: {lit!r})")

    # Transitive check: also walk one level of local (ai-engine/*.py) imports
    # from each patient-runtime file, in case a forbidden module is reached
    # indirectly through a helper file that isn't itself in the explicit list.
    visited: set[str] = set()
    to_visit = list(PATIENT_RUNTIME_FILES)
    while to_visit:
        filename = to_visit.pop()
        if filename in visited:
            continue
        visited.add(filename)
        py_file = CURRENT_DIR / filename
        if not py_file.exists():
            continue
        imports = collect_imports(py_file)
        bad_imports = imports & FORBIDDEN_MODULES
        if bad_imports:
            failures.append(f"{filename} (transitive): imports forbidden generic-template module(s): {sorted(bad_imports)}")
        for mod in imports:
            candidate = f"{mod}.py"
            if (CURRENT_DIR / candidate).exists() and candidate not in visited:
                to_visit.append(candidate)

    if failures:
        print("FAIL — patient reconstruction runtime still depends on generic-template code:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"PASS — none of {PATIENT_RUNTIME_FILES} (transitively) import a forbidden generic-template module, "
          f"call run_gnm_reconstruction, or reference a generic template asset path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
