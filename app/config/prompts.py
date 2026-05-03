"""Prompt loading utilities."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache prompts in memory
_PROMPT_CACHE: dict[str, str] = {}


def load_prompt(prompt_type: str, app_id: str | None = None) -> str:
    """Load system prompt from prompt.json based on promptType.
    
    Args:
        prompt_type: The type of prompt to load (e.g., 'agentSkill').
        app_id: Optional application ID for tenant-specific overrides (future).
        
    Returns:
        The content of the prompt, or a default string if not found.
    """
    global _PROMPT_CACHE
    
    if not _PROMPT_CACHE:
        prompt_path = Path("prompt.json")
        if prompt_path.exists():
            try:
                with open(prompt_path, encoding="utf-8") as f:
                    data = json.load(f)
                    for p in data.get("systemPrompts", []):
                        if "type" in p and "content" in p:
                            _PROMPT_CACHE[p["type"]] = p["content"]
            except Exception as e:
                logger.error(f"Failed to load prompt.json: {e}")
        else:
            logger.warning("prompt.json not found in working directory.")
            
    if prompt_type in _PROMPT_CACHE:
        return _PROMPT_CACHE[prompt_type]
        
    logger.warning(f"Prompt type '{prompt_type}' not found. Using empty fallback.")
    return "You are a helpful assistant."
