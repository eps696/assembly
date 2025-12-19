# Story Architect: Global Updates

## Task
Review last chapter's scenes, update globals, refine chapter summary.

## Process
1. **Analyze Scenes**:
   - Identify significant local actors/settings
   - Detect cliches used in scenes
   - Note narrative developments affecting chapter summary

2. **Update Globals**:
   - Promote important local actors to global_actors
   - Promote recurring local settings to locations
   - Add detected cliches to stop_cliches
   - Merge duplicates

3. **Refine Chapter**:
   - Update chapter content to reflect scene developments
   - Ensure consistency between outline and execution

## Output
```
{
  "global_settings": {
    "locations": ["Updated with promoted locations"],
    "primary_tension": "Updated if evolved",
    "stop_cliches": ["Including newly detected cliches"]
  },
  "global_actors": [
    {
      "name": "Actor name",
      "type": "character/creature/entity/force",
      "look": "Visual traits",
      "motivations": "Core drives",
      "background": "History",
      "objectives": "Goals and tensions"
    }
  ],
  "chapters": [
    {
      "chapter_number": N,
      "chapter_title": "Title",
      "chapter_content": "Updated content"
    }
  ]
}
```
Note: Only include sections with actual updates.
