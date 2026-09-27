from typing import Dict, List, Any


class StructuredPromptBuilder:
    DEFAULT_NEGATIVE_PROMPT = (
        "modern western clothing, jeans, t-shirt, inaccurate cultural symbols, distorted faces, "
        "extra limbs, extra fingers, mutated hands, bad anatomy, blurry, low resolution, watermark, "
        "signature, plastic skin, oversaturated, generic modern background, anime, cartoon"
    )

    def __init__(self, style_preset: str = "photorealistic"):
        self.style_preset = style_preset

    def build_prompt(
        self, 
        original_query: str, 
        english_query: str, 
        filtered_categories: Dict[str, List[str]],
        render_quality: str = "hyperrealistic 8k photograph, highly detailed masterwork"
    ) -> Dict[str, str]:

        def clean_list(items):
            return ", ".join([i.replace("Visual detail: ", "").strip() for i in items if i])

        attire = clean_list(filtered_categories.get("attire_jewelry", []))
        rituals = clean_list(filtered_categories.get("rituals_actions", []))
        architecture = clean_list(filtered_categories.get("architecture_setting", []))
        objects = clean_list(filtered_categories.get("objects_artifacts", []))
        atmosphere = clean_list(filtered_categories.get("environment_atmosphere", []))
        colors = clean_list(filtered_categories.get("colors_textures", []))
        regional = clean_list(filtered_categories.get("regional_characteristics", []))

        subject_part = f"Authentic cultural portrait: {english_query} ({original_query})" if original_query != english_query else f"Authentic cultural portrait: {english_query}"
        
        prompt_parts = [subject_part]

        if attire:
            prompt_parts.append(f"Attire & Jewelry: {attire}")
        if rituals or objects:
            ritual_obj_text = f"Ritual & Objects: {rituals}" if rituals else ""
            if objects:
                ritual_obj_text += f", holding {objects}" if ritual_obj_text else f"Objects: {objects}"
            prompt_parts.append(ritual_obj_text)
        if architecture:
            prompt_parts.append(f"Setting & Architecture: {architecture}")
        if atmosphere or colors:
            env_text = f"Lighting & Palette: {atmosphere}" if atmosphere else ""
            if colors:
                env_text += f", color scheme of {colors}" if env_text else f"Color Palette: {colors}"
            prompt_parts.append(env_text)
        if regional:
            prompt_parts.append(f"Regional Motif: {regional}")

        prompt_parts.append(f"Style: {render_quality}, rich cultural authenticity, 85mm portrait lens, f/1.8 depth of field, dramatic atmospheric illumination")

        full_prompt = ", ".join([p for p in prompt_parts if p])

        full_prompt = full_prompt.replace("  ", " ").strip()

        structured_blocks = {
            "Original Query": original_query,
            "Translated Concept": english_query,
            "Attire & Jewelry": attire or "Traditional regional attire",
            "Ritual & Pose": rituals or "Traditional posture",
            "Objects & Artifacts": objects or "Sacred ceremonial objects",
            "Setting & Architecture": architecture or "Authentic regional environment",
            "Lighting & Atmosphere": atmosphere or "Natural warm lighting",
            "Color Palette": colors or "Vibrant traditional colors",
            "Render Specs": render_quality
        }

        return {
            "full_prompt": full_prompt,
            "structured_blocks": structured_blocks,
            "negative_prompt": self.DEFAULT_NEGATIVE_PROMPT
        }
