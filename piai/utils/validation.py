"""
piai/utils/validation.py — Tool argument validation, type coercion, and StringEnum helper.
Mirrors validation.ts from the original.
"""
from __future__ import annotations

import weakref
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from piai.types import Tool, ToolCallContent

# WeakKeyDictionary cache: tool → compiled jsonschema validator
_validator_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def validate_tool_call(
    call: "ToolCallContent",
    tools: list["Tool"],
) -> dict[str, Any]:
    """
    Validate and coerce tool call arguments against the tool's JSON Schema.

    Returns the (possibly coerced) input dict.
    Raises ValueError if the tool is not found or validation fails.
    """
    tool = next((t for t in tools if t.name == call.name), None)
    if tool is None:
        raise ValueError(f"Unknown tool: {call.name!r}")

    coerced = _coerce_types(call.input, tool.input)

    try:
        import jsonschema  # type: ignore[import]
        jsonschema.validate(instance=coerced, schema=tool.input)
    except ImportError:
        pass  # validation is best-effort; jsonschema is a soft dependency
    except jsonschema.ValidationError as exc:
        raise ValueError(f"Tool {call.name!r} validation failed: {exc.message}") from exc

    return coerced


def _coerce_types(data: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively coerce values to match schema types.
    e.g. "123" → 123 for integer fields.
    """
    properties = schema.get("properties", {})
    result = dict(data)

    for key, prop_schema in properties.items():
        if key not in result:
            continue
        value = result[key]
        expected_type = prop_schema.get("type")

        if expected_type == "integer" and isinstance(value, str):
            try:
                result[key] = int(value)
            except ValueError:
                pass
        elif expected_type == "number" and isinstance(value, str):
            try:
                result[key] = float(value)
            except ValueError:
                pass
        elif expected_type == "boolean" and isinstance(value, str):
            result[key] = value.lower() in ("true", "1", "yes")
        elif expected_type == "object" and isinstance(value, dict):
            result[key] = _coerce_types(value, prop_schema)

    return result


class StringEnum:
    """
    Helper for Google Vertex compatibility.
    Google does not support anyOf/const for enums in tool schemas.

    Usage: converts {"enum": ["a", "b"]} → {"type": "STRING", "description": "One of: a, b"}
    Call StringEnum.convert(schema) before passing to Vertex.
    """

    @staticmethod
    def convert(schema: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively convert enum fields to Vertex-compatible STRING descriptions.
        Also converts anyOf/const patterns that Vertex doesn't support.
        """
        return _convert_for_vertex(schema)


def _convert_for_vertex(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return schema

    result = {}

    for key, value in schema.items():
        if key == "enum" and isinstance(value, list) and all(isinstance(v, str) for v in value):
            # Convert string enum to STRING with description
            result["type"] = "STRING"
            existing_desc = schema.get("description", "")
            enum_desc = f"One of: {', '.join(value)}"
            result["description"] = f"{existing_desc}. {enum_desc}".strip(". ") if existing_desc else enum_desc
            continue
        if key == "anyOf" and isinstance(value, list):
            # anyOf not supported by Vertex — pick first non-null option
            non_null = [v for v in value if v.get("type") != "null"]
            if non_null:
                return _convert_for_vertex(non_null[0])
            continue
        if key == "const":
            # const not supported — convert to a STRING description
            result["type"] = "STRING"
            result["description"] = f"Must be: {value}"
            continue
        if key == "$defs":
            # $defs not supported by Vertex — skip
            continue
        if isinstance(value, dict):
            result[key] = _convert_for_vertex(value)
        elif isinstance(value, list):
            result[key] = [_convert_for_vertex(v) if isinstance(v, dict) else v for v in value]
        else:
            result[key] = value

    return result
