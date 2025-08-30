from pathlib import Path

import pytest
from pytest import MonkeyPatch

from rag_service import main
from rag_service.interface_extractor import extract_public_interfaces

LANG_CASES = {
    "python": {
        "ext": "py",
        "code": (
            "class Example:\n"
            "    def public(self):\n        pass\n"
            "    def _private(self):\n        pass\n\n"
            "def standalone():\n    pass\n"
            "def _helper():\n    pass\n"
        ),
        "expected": [
            "class Example: [1:5]",
            "def public(self): [2:3]",
            "def standalone(): [7:8]",
        ],
        "unexpected": ["_private", "_helper"],
    },
    "kotlin": {
        "ext": "kt",
        "code": (
            "interface Sample<T> {\n"
            "    fun ifaceFun(t: T)\n"
            "}\n"
            "class Example<T> {\n"
            "    fun publicFun(t: T) {}\n"
            "    protected fun protectedFun() {}\n"
            "    internal fun internalFun() {}\n"
            "}\n"
            "object Single {\n"
            "    fun objectFun() {}\n"
            "    private fun hidden() {}\n"
            "}\n"
            "fun topLevel() {}\n"
        ),
        "expected": [
            "interface Sample<T> { [1:3]",
            "fun ifaceFun(t: T) [2:2]",
            "class Example<T> { [4:8]",
            "fun publicFun(t: T) { [5:5]",
            "object Single { [9:12]",
            "fun objectFun() { [10:10]",
            "fun topLevel() { [13:13]",
        ],
        "unexpected": ["protectedFun", "internalFun", "hidden"],
    },
    "java": {
        "ext": "java",
        "code": (
            "public interface Sample<T> {\n"
            "    T interfaceMethod(T param);\n"
            "    private void hidden();\n"
            "}\n"
            "public class Example<T> {\n"
            "    public void publicMethod() {}\n"
            "    protected void protectedMethod() {}\n"
            "    void packagePrivateMethod() {}\n"
            "    private void privateMethod() {}\n"
            "    public class Inner {\n"
            "        public void inner() {}\n"
            "        private void innerHidden() {}\n"
            "    }\n"
            "}\n"
        ),
        "expected": [
            "public interface Sample<T> { [1:4]",
            "T interfaceMethod(T param); [2:2]",
            "public class Example<T> { [5:14]",
            "public void publicMethod() { [6:6]",
            "public class Inner { [10:13]",
            "public void inner() { [11:11]",
        ],
        "unexpected": [
            "hidden",
            "protectedMethod",
            "packagePrivateMethod",
            "privateMethod",
            "innerHidden",
        ],
    },
    "javascript": {
        "ext": "js",
        "code": (
            "export default function defaultFunc() {}\n"
            "export function namedFunc() {}\n"
            "export const arrow = () => {}\n"
            "export const obj = {\n"
            "  method() {}\n"
            "}\n"
            "class Example {\n"
            "  publicMethod() {}\n"
            "  #privateMethod() {}\n"
            "}\n"
            "function topLevel() {}\n"
            "function _hidden() {}\n"
        ),
        "expected": [
            "function defaultFunc() { [1:1]",
            "function namedFunc() { [2:2]",
            "const arrow = [3:3]",
            "const obj = [4:6]",
            "method() { [5:5]",
            "class Example { [7:10]",
            "publicMethod() { [8:8]",
            "function topLevel() { [11:11]",
        ],
        "unexpected": ["#privateMethod", "_hidden"],
    },
    "typescript": {
        "ext": "ts",
        "code": (
            "export interface Sample<T> {\n"
            "  ifaceMethod(param: T): T;\n"
            "}\n"
            "export class Example<T> {\n"
            "  public publicMethod(): T { return; }\n"
            "  protected protectedMethod(): void {}\n"
            "  private privateMethod(): void {}\n"
            "  #privateMethod(): void {}\n"
            "}\n"
            "export function topLevel<T>(arg: T): T { return arg; }\n"
            "export const arrow = <T>(arg: T): T => arg;\n"
            "export const obj = {\n"
            "  method<T>(arg: T): T { return arg; }\n"
            "}\n"
            "function _hidden(): void {}\n"
        ),
        "expected": [
            "interface Sample<T> { [1:3]",
            "ifaceMethod(param: T): [2:2]",
            "class Example<T> { [4:9]",
            "public publicMethod(): T { [5:5]",
            "function topLevel<T>(arg: T): T { [10:10]",
            "const arrow = [11:11]",
            "const obj = [12:14]",
            "method<T>(arg: T): T { [13:13]",
        ],
        "unexpected": [
            "protectedMethod",
            "privateMethod",
            "#privateMethod",
            "_hidden",
        ],
    },
    "rust": {
        "ext": "rs",
        "code": (
            "pub struct PublicStruct {\n"
            "    pub field: i32,\n"
            "}\n"
            "struct PrivateStruct {\n"
            "    field: i32,\n"
            "}\n"
            "pub enum PublicEnum {\n"
            "    A,\n"
            "    B,\n"
            "}\n"
            "enum PrivateEnum {\n"
            "    A,\n"
            "}\n"
            "impl PublicStruct {\n"
            "    pub fn new() -> Self {\n"
            "        Self { field: 0 }\n"
            "    }\n"
            "    fn hidden() {}\n"
            "}\n"
        ),
        "expected": [
            "pub struct PublicStruct { [1:3]",
            "pub enum PublicEnum { [7:10]",
            "pub fn new() -> Self { [15:17]",
        ],
        "unexpected": ["PrivateStruct", "PrivateEnum", "hidden"],
    },
    "go": {
        "ext": "go",
        "code": (
            "package sample\n\n"
            "type PublicStruct struct {\n"
            "    Field int\n"
            "}\n\n"
            "type privateStruct struct {\n"
            "    field int\n"
            "}\n\n"
            "func PublicFunc() {}\n\n"
            "func privateFunc() {}\n"
        ),
        "expected": [
            "type PublicStruct struct { [3:5]",
            "func PublicFunc() { [11:11]",
        ],
        "unexpected": ["privateStruct", "privateFunc"],
    },
    "swift": {
        "ext": "swift",
        "code": (
            "public protocol Sample {\n"
            "    func protoFunc()\n"
            "    private func hidden()\n"
            "}\n"
            "public class Example {\n"
            "    public func publicMethod() {}\n"
            "    internal func internalMethod() {}\n"
            "    private func privateMethod() {}\n"
            "}\n"
            "public struct MyStruct {\n"
            "    public var field: Int\n"
            "    var hidden: Int\n"
            "}\n"
            "public func topLevel() {}\n"
            "func internalFunc() {}\n"
        ),
        "expected": [
            "public protocol Sample { [1:4]",
            "func protoFunc() [2:2]",
            "public class Example { [5:9]",
            "public func publicMethod() { [6:6]",
            "public struct MyStruct { [10:13]",
            "public var field: [11:11]",
            "public func topLevel() { [14:14]",
        ],
        "unexpected": [
            "hidden",
            "internalMethod",
            "privateMethod",
            "internalFunc",
        ],
    },
}


@pytest.mark.parametrize("lang", LANG_CASES)
def test_extract_public_interfaces_languages(tmp_path: Path, lang: str) -> None:
    """It extracts public interfaces for supported languages."""

    case = LANG_CASES[lang]
    file_path = tmp_path / f"sample.{case['ext']}"
    file_path.write_text(case["code"])
    interfaces = extract_public_interfaces(file_path, lang)
    for expected in case["expected"]:
        assert expected in interfaces
    for name in case["unexpected"]:
        assert all(name not in s for s in interfaces)


def test_extract_public_interfaces_invalid_language(tmp_path: Path) -> None:
    """Unsupported language yields no interfaces."""

    fp = tmp_path / "sample.foo"
    fp.write_text("func main() {}")
    assert extract_public_interfaces(fp, "foo") == []


def test_extract_public_interfaces_syntax_error(tmp_path: Path) -> None:
    """Syntax errors result in empty interface lists."""

    fp = tmp_path / "broken.py"
    fp.write_text("def broken(:\n    pass")
    assert extract_public_interfaces(fp, "python") == []


def test_query_endpoint_interfaces(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Query endpoint returns interface data when requested."""

    code = "def foo():\n    pass\n"
    fp = tmp_path / "test.py"
    fp.write_text(code)

    class Node:
        """Mock node mimicking a vector store result."""

        def __init__(self, metadata: dict) -> None:
            self.metadata = metadata

        def get_content(self) -> str:
            return code

    class Result:
        """Container for a mocked node result."""

        def __init__(self, metadata: dict) -> None:
            self.node = Node(metadata)
            self.score = 1.0

    class Retriever:
        """Simple retriever returning a fixed result."""

        def retrieve(self, q: str) -> list[Result]:
            return [Result({"file_path": str(fp), "lang": "python"})]

    def fake_build_query_engine(cfg, qdrant, llama, collection_prefix):
        return Retriever()

    monkeypatch.setattr(main, "build_query_engine", fake_build_query_engine)
    main.CONFIG = type("Cfg", (), {"features": type("F", (), {})(), "qdrant": object()})()
    main.QDRANT = object()
    main.LLAMA = object()

    resp = main.query_endpoint(
        main.QueryRequest(root_path=str(tmp_path), q="test", interfaces=True)
    )
    assert resp["items"][0]["interfaces"] == ["def foo(): [1:2]"]
