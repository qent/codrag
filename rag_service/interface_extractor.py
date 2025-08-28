from __future__ import annotations

from pathlib import Path
from typing import List

from tree_sitter import Node
from tree_sitter_language_pack import get_parser


PUBLIC_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "class_definition",
    "class_declaration",
    "interface_declaration",
    "struct_declaration",
    "enum_declaration",
}


def _signature_from_node(node: Node, source: bytes) -> str:
    line = source[node.start_byte : node.end_byte].splitlines()[0]
    line = line.decode("utf-8", errors="ignore").strip()
    for sep in (":", "{"):
        if sep in line:
            line = f"{line.split(sep)[0]}{sep}"
            break
    return line


def _is_public(name: str, signature: str) -> bool:
    if name.startswith("_"):
        return False
    if "private" in signature:
        return False
    return True


def extract_public_interfaces(file_path: Path, lang: str | None) -> List[str]:
    """Return a list of public interfaces found in ``file_path``.

    Each interface signature is suffixed with ``[first_line:last_line]`` to
    indicate its location in the file. The function uses ``tree-sitter`` to
    parse the file using the provided language identifier. If the language is
    unsupported or parsing fails, an empty list is returned.
    """

    try:
        parser = get_parser(lang or "")
    except Exception:
        return []
    source = file_path.read_bytes()
    tree = parser.parse(source)

    signatures: List[str] = []

    def visit(node: Node) -> None:
        if node.type in PUBLIC_NODE_TYPES:
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore")
                sig = _signature_from_node(node, source)
                if _is_public(name, sig):
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    signatures.append(f"{sig} [{start_line}:{end_line}]")
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return signatures
