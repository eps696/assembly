# Assembly: Agentic Narratives

<p align='center'><img src='_out/035-f-306.jpg' /></p>

This is a barebone setup with basic scaffolding for agentic narrative explorations with continuous, semi-autonomous multimedia content generation. The prompts are intentionally minimal and should be enhanced and adapted to specific creative tasks.

The system uses LLM agents to orchestrate narrative generation, which feeds into a pipeline for image, video, and audio generation. 
Visual generation uses [Runware.ai] cloud service for its optimal price/performance ratio. Audio generation uses [Chatterbox] text-to-speech and [MMAudio] video-to-audio methods (their codebases included in this repo).

## Backends

### Agent Types

- **LMStudio Agent** (`agt_llm.py`) - OpenAI-compatible API for local models (Llama, Qwen, Mistral, etc.) via [LMStudio] or cloud (GPT-5, GPT-4o-mini; API key required). Supports tool calling with automatic malformed JSON recovery.
- **Google ADK Agent** (`agt_adk.py`) - Native Gemini integration via Google's [Agent Development Kit] (API key required). Supports Google Search as a built-in tool.

### Media Generation

- **Text-to-Image (T2I)** via [Runware API] (cloud, API key required) - Flux, Imagen, HiDream models. 
- **Image-to-Video (I2V)** via [Runware API] (cloud, API key required) - Seedance, Veo models
- **Text-to-Speech (TTS)** via [ChatterBox] (local) - Multi-speaker neural TTS with voice cloning
- **Audio Mixing** via [MMAudio] (local) - Video-to-audio generation

## Internal Features

**Async State Management** (`base.py`)
Asynchronous processing and thread-safe state handling with JSON persistence. Supports incremental merging of agent outputs and automatic checkpointing after each step. The `StoryState` class handles concurrent access and ensures data consistency across async operations.

**QA Evaluation Loop** (`--evals N`)
Each agent call can be followed by an evaluation pass using a separate LLM call. The evaluator scores the output and provides feedback; if not approved, the agent retries with the feedback incorporated. This significantly improves output quality at the cost of additional API calls.

**Built-in Agentic Tools** (`tools.py`)
- `brave_search` - Web search via Brave Search API for grounding with current information
- `fetch_url_content` - URL fetching with automatic HTML-to-markdown conversion
- `@tool` decorator - Auto-generates OpenAI-compatible tool schemas from function signatures and docstrings

**Sequential Thinking Tool** (`tool_think.py`)
Provides structured chain-of-thought reasoning for smaller LLMs that lack native reasoning capabilities. The tool exposes a `sequential_thinking` function that tracks thought sequences, supports branching (exploring alternatives), and revision (reconsidering previous thoughts). The tool auto-manages numbering since small models often lose track of sequence counts.

**Malformed JSON Recovery** (`agt_llm.py`)
The LMStudio agent automatically recovers from malformed tool calls - common with smaller models that struggle with strict JSON formatting. Invalid responses trigger automatic retries with corrected parsing.

## Workflows

### Story Mode

```
# Text-only generation
python src/author.py -txt

# Generation with sound and visuals
python src/author.py -vt runware -imod "bfl:5@1" -isz 1344-752 -vmod "bytedance:2@2" -vsz 864-480 -fps 24 -iref 8

# Resume from saved state and config
python src/author.py -json _out/log.json -arg _out/config.txt 
```

Hierarchical content generation for narrative storytelling (see `data/prompts/story`):
- **arch-init** - Initialize story structure, settings, and characters (+ auto-generate reference images for visual consistency)
- **writ-chap** - Generate chapter outlines from the overall narrative arc
- **writ-scen** - Create detailed scenes within each chapter
- **writ-frag** - Write narrative fragments (the actual prose)
- **writ-voc** - Generate voice-over scripts (+ auto-convert to speech via TTS)
- **writ-vis** - Generate visual descriptions (+ auto-create images (T2I), animate to video (I2V), mix with audio)
- **arch-upd** - Update global story state based on generated content

**Story Schema**
```
global_settings    - Title, genre, themes, narrative constraints
global_actors      - Character definitions with visual descriptions
chapters           - Chapter outlines and summaries
scenes             - Scene breakdowns within chapters
fragments          - Narrative prose fragments
voices             - Voice-over scripts with speaker assignments
visuals            - Visual descriptions and generation metadata
```

### Chat Mode

```
python src/chat.py ....
```

Linear multi-persona discussions (see `data/chat/story`):
- **arch-init** - Create debate personas with distinct perspectives, styles, and voice profiles (+ auto-generate ref images)
- **writ-frag** - Generate each persona's elaborated inner thoughts for their turn
- **writ-voc** - Condense thoughts into short spoken remarks (+ auto-convert to speech via TTS)
- **writ-vis** - Generate visual representation of the current speaker/moment (+ auto-create images/videos with audio)
- **arch-upd** - Update persona states and debate direction based on progress

**Chat Schema**

```
global_settings    - Topic, constraints, focus areas, debate rules
personas           - Character definitions with roles, styles, and speaker IDs
fragments          - Elaborated thoughts for each turn
voices             - Condensed spoken remarks
visuals            - Visual descriptions for each moment
```

See `data/schema-story.json` and `data/schema-chat.json` for complete structure.
Both workflows save state to `log.json` after each step, enabling resume from any point.

## Other Usage

```
# Use local LLM
... -a lms -tmod openai/gpt-oss-20b -lh localhost

# Use OpenAI cloud
... -a lms -tmod gpt-4o-mini

# Use Gemini via Google ADK
... -a adk -tmod gemini-2.5-flash

# Extract discussion transcript
python src/readlog.py -i _out/log.json -o discussion.txt

```

## Key Arguments

- `-a/--agent` - Agent type: `lms` (LMStudio/OpenAI) or `adk` (Google ADK)
- `-tmod/--txt_model` - LLM model name
- `-lh/--llm_host` - LMStudio server host
- `-json/--load_json` - Resume from JSON state file

- `-vt/--vis_type` - Visual backend: runware, comfy, wan, walk
- `-imod/--img_model` - Image model (e.g., bfl:5@1 for Flux 2 Pro)
- `-vmod/--vid_model` - Video model (e.g., bytedance:2@2 for Seedance Pro Fast)
- `-isz/--img_size` - Image dimensions (e.g., 1344-752)
- `-vsz/--vid_size` - Video dimensions (e.g., 864-480)
- `-fps` - Video framerate
- `-iref/--img_refs` - Number of reference images for consistency

- `-txt/--txtonly` - Text-only mode, skip media
- `-o/--out_dir` - Output directory (default: _out)
- `-v/--verbose` - Verbose output

## Files

```
assembly/
├── src/                    # Core library
│   ├── author.py           # Story generation pipeline
│   ├── chat.py             # Interactive chat/debate pipeline
│   ├── base.py             # Infrastructure (MediaGen, StoryState, PromptMan, etc.)
│   ├── agt_llm.py          # LMStudio/OpenAI agent with tool calling
│   ├── agt_adk.py          # Google ADK (Gemini) agent
│   ├── tools.py            # Tool system
│   ├── tool_think.py       # Sequential thinking tool for small LLMs
│   ├── api_runware.py      # Runware cloud integration (T2I, I2V)
│   ├── sound.py            # Audio mixing (MMAudio)
│   ├── tts.py              # Text-to-speech (ChatterBox)
│   ├── readlog.py          # Extract discussion from log.json
│   └── util.py             # Utilities
├── _in/                    # Input documents (source materials)
├── _out/                   # Generated outputs
│   └── join_vids.bat       # FFmpeg video concatenation
├── data/                   # JSON schemas, templates, databases, etc.
│   └── prompts/            # Markdown prompt templates
│       ├── story/          # Story mode prompts
│       └── chat/           # Chat/debate mode prompts
└── au.bat                  # Base wrapper script
```

[LMStudio]: <https://lmstudio.ai/>
[Agent Development Kit]: <https://github.com/google/adk-python>
[Runware.ai]: <https://runware.ai>
[Chatterbox]: <https://github.com/resemble-ai/chatterbox>
[MMAudio]: <https://github.com/hkchengrex/MMAudio>
