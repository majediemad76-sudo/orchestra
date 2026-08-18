"""One Pydantic model in, three vendor schema dialects out.

Asking a model to "reply with JSON only" is a request; a schema enforced at the
API layer is a guarantee. Every role in this system depends on the guarantee,
which makes this module the single place where a vendor's schema quirks are
allowed to exist. The three dialects share nothing but their ancestry:

  * Anthropic has no ``response_format``. The schema rides in a tool's
    ``input_schema`` and the tool is then forced -- see ``anthropic_tool``.
  * xAI speaks the OpenAI ``json_schema`` dialect, whose ``strict`` mode
    imposes two rules that JSON Schema itself does not -- see ``to_xai_schema``.
  * Google has neither, and uses an upper-cased near-JSON-Schema of its own
    with a dedicated ``nullable`` flag -- see ``to_gemini_schema``.

Pydantic's own output is the input to all three, and it is never directly
usable: it factors nested models out into ``$defs`` and references them by
``$ref``, which none of the three vendors resolve. ``resolve_refs`` runs first
in every path here.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel

# Strict mode rejects a schema outright if it carries a keyword outside this
# set -- it fails closed, and the error names the schema rather than the
# offending keyword. Allow-list what is known to pass; everything else is
# stripped and restated in prose (see _constraint_prose) so the intent reaches
# the model even though the validator no longer enforces it.
_XAI_KEEP = {
    "type",
    "description",
    "properties",
    "required",
    "items",
    "enum",
    "additionalProperties",
    "anyOf",
}

# Gemini ignores unknown keys rather than erroring, which is worse than failing
# loudly: a dropped constraint is invisible until the output is wrong. Keep the
# allow-list narrow and deliberate.
_GEMINI_KEEP = {
    "type",
    "description",
    "properties",
    "required",
    "items",
    "enum",
    "format",
    "nullable",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "propertyOrdering",
}

# Bounds a dialect cannot express are not silently discarded -- they are
# appended to the field description, where they still steer the model.
_CONSTRAINT_KEYS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
)

_MAX_REF_DEPTH = 20


def _lookup(root: Dict[str, Any], pointer: str) -> Dict[str, Any]:
    """Follow a local JSON pointer, e.g. ``#/$defs/Question``.

    Remote pointers are refused rather than fetched: a schema that reaches out
    over the network at build time is a failure mode nobody wants to debug.
    """
    if not pointer.startswith("#/"):
        raise ValueError(f"only local $ref is supported, got: {pointer}")
    node: Any = root
    for part in pointer[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        node = node[part]
    return node


def resolve_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline every ``$ref`` and drop the ``$defs`` block.

    Pydantic emits a ``$ref`` for every nested model. Anthropic and xAI reject
    the reference, and Gemini quietly ignores it -- so the tree is flattened
    once, here, instead of three times downstream.

    A self-referential model (``Node.children: list[Node]``) would expand
    forever, so depth is capped and the node past the cap degrades to an
    unconstrained object. None of the current schemas recurse; the cap exists
    so that adding one later fails visibly rather than by hanging.
    """
    root = copy.deepcopy(schema)

    def walk(node: Any, depth: int) -> Any:
        if isinstance(node, list):
            return [walk(item, depth) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            if depth >= _MAX_REF_DEPTH:
                return {"type": "object"}
            target = copy.deepcopy(_lookup(root, node["$ref"]))
            # A field's own description sits beside the $ref, while the
            # target carries the nested model's. The field-level one is more
            # specific, so it wins.
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            target.update(siblings)
            return walk(target, depth + 1)
        return {k: walk(v, depth) for k, v in node.items() if k != "$defs"}

    resolved = walk(root, 0)
    resolved.pop("$defs", None)
    return resolved


def _split_nullable(node: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Unwrap ``Optional[X]`` into its inner schema plus a nullable flag.

    Pydantic renders ``Optional[X]`` as ``anyOf: [X, {"type": "null"}]``.
    Neither xAI's strict mode nor Gemini accepts that shape, and both need the
    nullability expressed in their own way -- so it is extracted here once and
    re-applied per dialect.

    A genuine multi-branch union (three or more real types) is left untouched:
    collapsing it would silently narrow the contract. It is returned as-is with
    its null-ness reported, and the caller's allow-list decides its fate.
    """
    if "anyOf" not in node:
        return node, False
    branches = node["anyOf"]
    non_null = [b for b in branches if b.get("type") != "null"]
    has_null = len(non_null) != len(branches)
    if not has_null or len(non_null) != 1:
        return node, has_null
    merged = copy.deepcopy(non_null[0])
    for key, value in node.items():
        if key != "anyOf" and key not in merged:
            merged[key] = value
    return merged, True


def _constraint_prose(node: Dict[str, Any]) -> str:
    parts = [f"{k}={node[k]}" for k in _CONSTRAINT_KEYS if k in node]
    return f" (constraints: {', '.join(parts)})" if parts else ""


def to_anthropic_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Schema for an Anthropic tool's ``input_schema``.

    The gentlest of the three dialects -- plain JSON Schema, extra keywords
    tolerated -- so nothing is stripped. Only the ``$ref`` indirection has to
    go, and a top-level ``type`` is asserted because the API rejects a tool
    whose input schema is not an object.
    """
    schema = resolve_refs(model.model_json_schema())
    schema.setdefault("type", "object")
    return schema


def anthropic_tool(model: Type[BaseModel], name: str = "emit_result", description: str = "") -> Dict[str, Any]:
    """Build the tool that Claude will be forced to call.

    This is Anthropic's substitute for ``response_format``, and it is a
    stronger constraint rather than a weaker one: paired with
    ``tool_choice={"type": "tool", ...}`` at the call site, prose is not an
    available move for the model. Sampling is constrained to arguments that
    satisfy ``input_schema``, so "the model ignored the format instruction"
    stops being a failure mode.

    The tool is a pure output channel -- nothing executes when it is called.
    """
    return {
        "name": name,
        "description": description or f"Emit the result as a {model.__name__} object.",
        "input_schema": to_anthropic_schema(model),
    }


def to_xai_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Schema for xAI's ``response_format.json_schema``, strict mode on.

    Strict mode is what makes the output trustworthy, and it adds two rules
    that plain JSON Schema does not have:

    1. Every object must carry ``additionalProperties: false``. Omit it on any
       nested object -- easy to miss, since it is not where the model's
       attention is -- and the whole request is rejected.
    2. Every property must appear in ``required``. There is no such thing as an
       absent field, so Python-optional fields are widened to accept ``null``
       instead (``["string", "null"]``). The field is then always present and
       explicitly empty, which is also easier to read downstream than a key
       that may or may not exist.

    Both rules are applied here rather than trusted to the schema author.
    """

    def convert(node: Dict[str, Any]) -> Dict[str, Any]:
        node, nullable = _split_nullable(node)
        prose = _constraint_prose(node)
        out: Dict[str, Any] = {k: v for k, v in node.items() if k in _XAI_KEEP}
        if prose:
            out["description"] = (out.get("description", "") + prose).strip()

        node_type = out.get("type")
        if node_type == "object" or "properties" in node:
            props = {k: convert(v) for k, v in node.get("properties", {}).items()}
            out["type"] = "object"
            out["properties"] = props
            # Rule 2: required lists every property, without exception.
            # Optionality is re-expressed as a nullable type below.
            out["required"] = list(props.keys())
            out["additionalProperties"] = False
            optional = [k for k in props if k not in node.get("required", [])]
            for key in optional:
                _make_xai_nullable(props[key])
        elif node_type == "array" and "items" in node:
            out["items"] = convert(node["items"])

        if nullable:
            _make_xai_nullable(out)
        return out

    schema = convert(resolve_refs(model.model_json_schema()))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": schema,
        },
    }


def _make_xai_nullable(node: Dict[str, Any]) -> None:
    """Widen a node's type to admit ``null``, in place.

    A node with no ``type`` at all (a bare enum, say) is left alone: strict
    mode already constrains it by enumeration, and inventing a type here would
    widen the contract rather than the nullability.
    """
    node_type = node.get("type")
    if node_type is None:
        return
    if isinstance(node_type, list):
        if "null" not in node_type:
            node_type.append("null")
    elif node_type != "null":
        node["type"] = [node_type, "null"]


def to_gemini_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Schema for Google's ``generationConfig.responseSchema``.

    Gemini's dialect looks like JSON Schema and is not:

    * Type names are upper-case enum values -- ``STRING``, ``OBJECT``,
      ``ARRAY``, ``INTEGER``, ``BOOLEAN`` -- not the lower-case JSON Schema
      strings Pydantic emits.
    * There is no union type. Nullability is a sibling flag, ``nullable:
      true``, so every ``Optional[X]`` and every field carrying a default is
      marked rather than type-widened.
    * Unknown keys are ignored silently, so a constraint that does not survive
      the allow-list disappears without an error. That is why the allow-list is
      explicit and why ``propertyOrdering`` is set: field order is otherwise
      unspecified, and a stable order keeps logged output diffable.

    Note the two distinct sources of nullability: the ``anyOf``-with-null shape
    from ``Optional[X]``, and mere absence from the parent's ``required`` list
    (a field with a default). Both end up as ``nullable: true``.
    """

    def convert(node: Dict[str, Any], nullable: bool = False) -> Dict[str, Any]:
        node, from_union = _split_nullable(node)
        nullable = nullable or from_union
        out: Dict[str, Any] = {k: v for k, v in node.items() if k in _GEMINI_KEEP}

        node_type = node.get("type")
        if isinstance(node_type, list):
            non_null = [t for t in node_type if t != "null"]
            nullable = nullable or len(non_null) != len(node_type)
            node_type = non_null[0] if non_null else "string"
        # Pydantic omits "type" where it is inferable; Gemini requires it.
        if node_type is None and "properties" in node:
            node_type = "object"
        if node_type is None and "enum" in node:
            node_type = "string"
        if node_type is not None:
            out["type"] = str(node_type).upper()

        if "properties" in node:
            required = node.get("required", [])
            # Absent from "required" means the field has a Python default,
            # which Gemini can only express as nullable.
            out["properties"] = {
                k: convert(v, nullable=k not in required)
                for k, v in node["properties"].items()
            }
            out["propertyOrdering"] = list(node["properties"].keys())
            if required:
                out["required"] = list(required)
            else:
                out.pop("required", None)
        if node_type == "array" and "items" in node:
            out["items"] = convert(node["items"])

        if nullable:
            out["nullable"] = True
        return out

    return convert(resolve_refs(model.model_json_schema()))


def json_hint(model: Type[BaseModel]) -> str:
    """Render the schema as prose for the system prompt.

    Belt to the API's braces. The schema already makes malformed output
    impossible, but a model that knows what each field is *for* fills it in
    better than one merely prevented from getting it wrong -- descriptions do
    not always survive into every dialect's validator.
    """
    schema = resolve_refs(model.model_json_schema())
    lines: List[str] = []
    required = set(schema.get("required", []))
    for name, prop in schema.get("properties", {}).items():
        prop, nullable = _split_nullable(prop)
        kind = prop.get("type", "object")
        if "enum" in prop:
            kind = " | ".join(repr(v) for v in prop["enum"])
        flag = "" if name in required and not nullable else " (optional)"
        desc = prop.get("description", "")
        lines.append(f"- {name}: {kind}{flag}{(' — ' + desc) if desc else ''}")
    return "\n".join(lines)
