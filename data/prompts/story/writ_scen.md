# Story Writer: Scene Development

## Task
Transform last chapter outline into 6-10 detailed, immersive scenes. Avoid cliches (shimmering holograms, fantasy heroes, etc.)

## Process
- Break chapter into 6-10 scenes with clear transitions
- Transform summary into vivid, moment-by-moment storytelling
- Show events through action and communication, not exposition
- Create sensory-rich descriptions
- Introduce local settings and actors as needed (contained within chapter)

## Scene Elements
- **Local Settings**: Specific locations within world
- **Local Actors**: Minor characters/entities serving the scene
- **Local Details**: Objects, cultural elements enriching world
- Clear purpose and outcome per scene
- Varied pacing and length

## Requirements
- Copy full visual traits from globals for all actors/settings
- Include detailed visual traits for new local elements
- Maintain consistency with globals and previous chapters
- Scenes must completely cover chapter outline
- Each scene advances story meaningfully
- You MUST follow the Output format precisely, always filling up all fields

## Output
```
{
  "scenes": [
    {
      "chapter_number": 1,
      "chapter_title": "Title",
      "scene_number": 1,
      "global_setting": "Location: Specific setting from world",
      "global_actors": [
        {"name": "Actor", "look": "Full visual from global_actors.look"}
      ],
      "local_settings": ["New location: visual description"],
      "local_actors": ["New actors (if any) with visual traits"],
      "scene_content": "Detailed narrative with communication, action, description",
      "scene_purpose": "What this accomplishes"
    }
  ]
}
```
