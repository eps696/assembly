#!/usr/bin/env python
"""Extract voices from log.json as plain text discussion."""

import json
import argparse
import textwrap
from pathlib import Path

def load_state(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_discussion(state, msg_type, wrap_width=None):
    """Extract pieces as plain text discussion."""
    pieces = state.get(msg_type, [])
    if not pieces:
        return f"No {msg_type} found in state."

    lines = []
    for v in pieces:
        name = v.get('persona_name', 'Unknown')
        content = v.get('content', '')
        frag_num = v.get('fragment_number', '?')

        if wrap_width:
            wrapped = textwrap.fill(content, width=wrap_width, initial_indent='  ', subsequent_indent='  ')
            lines.append(f"[{frag_num}] {name}:\n{wrapped}\n")
        else:
            lines.append(f"[{frag_num}] {name}: {content}\n")

    return '\n'.join(lines)

def list_personas(state):
    """List all personas with their roles."""
    personas = state.get('personas', [])
    if not personas:
        return "No personas found."

    lines = ["Personas:"]
    for p in personas:
        name = p.get('name', 'Unknown')
        role = p.get('role', '')
        speaker = p.get('speaker', '')
        lines.append(f"  - {name} ({role}) [{speaker}]")
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description='Extract voices from log.json as discussion')
    parser.add_argument('-i', '--input', default='log.json', help='Path to log.json')
    parser.add_argument('-t', '--type', default='say', help='fragments (thoughts), voices (sayings), or visuals')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('-w', '--wrap', default=100, type=int, help='Wrap text at width (0=no wrap, default: 100)')
    parser.add_argument('-p', '--personas', action='store_true', help='List personas')
    args = parser.parse_args()

    extypes = {'frag':'fragments', 'say':'voices', 'voc':'voices', 'voi':'voices', 'vis':'visuals'}
    
    json_path = Path(args.input)
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return 1

    state = load_state(json_path)
    wrap_width = args.wrap if args.wrap > 0 else None
    msg_type = 'fragments' if 'fra' in args.type else 'visuals' if 'vis' in args.type else 'voices'

    output_parts = []
    settings = state.get('global_settings', {})
    topic = settings.get('topic', 'No topic')
    output_parts.append(f"Topic: {topic}\n")
    if args.personas:
        output_parts.append(list_personas(state))
    output_parts.append(extract_discussion(state, msg_type, wrap_width))
    result = '\n'.join(output_parts)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Written to {args.output}")
    else:
        print(result)


if __name__ == '__main__':
    main()
