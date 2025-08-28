from __future__ import annotations

from pathlib import Path
from typing import List

from tree_sitter import Node
from tree_sitter_language_pack import get_parser


PUBLIC_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "function_item",
    "method_definition",
    "method_declaration",
    "method_signature",
    "class_definition",
    "class_declaration",
    "interface_declaration",
    "object_declaration",
    "struct_declaration",
    "struct_item",
    "enum_declaration",
    "enum_item",
    "type_declaration",
    "lexical_declaration",
    "variable_declaration",
}

NAME_NODE_TYPES = {
    "identifier",
    "type_identifier",
    "simple_identifier",
}


def _signature_from_node(node: Node, source: bytes) -> str:
    line = source[node.start_byte : node.end_byte].splitlines()[0]
    line = line.decode("utf-8", errors="ignore").strip()
    for sep in ("=", "{"):
        if sep in line:
            return f"{line.split(sep)[0]}{sep}"
    if ":" in line:
        idx = line.rfind(":")
        if idx > line.rfind(")") and idx > line.rfind("}"):
            return f"{line[:idx + 1]}"
    return line


def _is_public(name: str, signature: str, node: Node, lang: str) -> bool:
    """Determine if the given ``name`` and ``signature`` are public."""

    if name.startswith("_") or name.startswith("#"):
        return False
    lower = signature.lower()
    if any(mod in lower for mod in ("private", "protected", "internal")):
        return False
    if lang == "java":
        if "public" not in lower:
            ancestor = node.parent
            found_interface = False
            while ancestor is not None:
                if ancestor.type == "interface_declaration":
                    found_interface = True
                    break
                ancestor = ancestor.parent
            if not found_interface:
                return False
    if lang == "rust":
        if not lower.startswith("pub"):
            return False
    if lang == "go":
        if name[:1].islower():
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
    if tree.root_node.has_error:
        return []

    signatures: List[str] = []

    def visit(node: Node) -> None:
        if node.type in PUBLIC_NODE_TYPES:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                def find_name(n: Node) -> Node | None:
                    for child in n.children:
                        if child.type in NAME_NODE_TYPES:
                            return child
                        found = find_name(child)
                        if found is not None:
                            return found
                    return None

                name_node = find_name(node)
            if name_node is not None:
                name = source[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore")
                sig = _signature_from_node(node, source)
                if _is_public(name, sig, node, lang or ""):
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    signatures.append(f"{sig} [{start_line}:{end_line}]")
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return signatures
