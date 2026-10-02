from typing import Dict, List, Any


class StructuredPromptBuilder:
    DEFAULT_NEGATIVE_PROMPT = (
        "modern western clothing, jeans, t-shirt, inaccurate cultural symbols, distorted faces, "
        "extra limbs, extra fingers, mutated hands, bad anatomy, blurry, low resolution, watermark, "
        "signature, plastic skin, oversaturated, generic modern background, anime, cartoon, "
        "missing main deity idol, obscured central idol, deity cut off from frame, out of focus main event, "
        "small background deity, distorted deity arms"
    )

    EVENT_FOCAL_MAP = {
        "durga": "majestic ten-armed Goddess Durga idol (Durga Mata ji) sculpted shrine centerpiece",
        "chhath": "glowing setting sun god (Surya Dev) reflecting over river ghat offering centerpiece",
        "pongal": "decorated earthen Pongal pot boiling over with rice centerpiece",
        "gudi": "hoisted Gudi flag draped in silk with silver kalash pot centerpiece",
        "onam": "grand multi-colored Pookkalam floral carpet centerpiece",
        "kathakali": "pacca green-masked Kathakali performer beside brass nilavilakku oil lamp centerpiece"
    }

    def __init__(self, style_preset: str = "photorealistic"):
        self.style_preset = style_preset

    def _determine_focal_centerpiece(self, original_query: str, english_query: str, extracted_focal: str) -> str:
        if extracted_focal:
            return extracted_focal

        combined = f"{original_query} {english_query}".lower()
        for key, centerpiece in self.EVENT_FOCAL_MAP.items():
            if key in combined:
                return centerpiece
        
        return ""

    def build_prompt(
        self, 
        original_query: str, 
        english_query: str, 
        filtered_categories: Dict[str, List[str]],
        render_quality: str = "hyperrealistic 8k photograph, highly detailed masterwork"
    ) -> Dict[str, str]:

        def clean_list(items):
            return ", ".join([i.replace("Visual detail: ", "").strip() for i in items if i])

        focal_extracted = clean_list(filtered_categories.get("focal_deity_centerpiece", []))
        attire = clean_list(filtered_categories.get("attire_jewelry", []))
        rituals = clean_list(filtered_categories.get("rituals_actions", []))
        architecture = clean_list(filtered_categories.get("architecture_setting", []))
        objects = clean_list(filtered_categories.get("objects_artifacts", []))
        atmosphere = clean_list(filtered_categories.get("environment_atmosphere", []))
        colors = clean_list(filtered_categories.get("colors_textures", []))
        regional = clean_list(filtered_categories.get("regional_characteristics", []))

        focal_centerpiece = self._determine_focal_centerpiece(original_query, english_query, focal_extracted)

        prompt_parts = []

        # 1. Main Focal Centerpiece with Attention Weighting (1.3)
        if focal_centerpiece:
            prompt_parts.append(f"Main Focal Centerpiece: ({focal_centerpiece}:1.3)")

        # 2. Foreground Subject
        subject_part = f"Foreground Subject: {english_query} ({original_query})" if original_query != english_query else f"Foreground Subject: {english_query}"
        prompt_parts.append(subject_part)

        # 3. Spatial Composition Directive
        if focal_centerpiece:
            prompt_parts.append(
                f"Spatial Composition: wide-angle cinematic shot clearly framing both the {focal_centerpiece} prominently in the center background and the subject in the foreground"
            )

        # 4. Secondary Cultural Visual Details
        if attire:
            prompt_parts.append(f"Attire & Jewelry: {attire}")
        if rituals or objects:
            ritual_obj_text = f"Ritual & Actions: {rituals}" if rituals else ""
            if objects:
                ritual_obj_text += f", holding {objects}" if ritual_obj_text else f"Objects: {objects}"
            prompt_parts.append(ritual_obj_text)
        if architecture:
            prompt_parts.append(f"Setting & Architecture: {architecture}")
        if atmosphere or colors:
            env_text = f"Lighting & Atmosphere: {atmosphere}" if atmosphere else ""
            if colors:
                env_text += f", color scheme of {colors}" if env_text else f"Color Palette: {colors}"
            prompt_parts.append(env_text)
        if regional:
            prompt_parts.append(f"Regional Motif: {regional}")

        # 5. Render Quality & Style Specs
        prompt_parts.append(f"Style: {render_quality}, rich cultural authenticity, balanced depth of field, dramatic illumination")

        full_prompt = ", ".join([p for p in prompt_parts if p])
        full_prompt = full_prompt.replace("  ", " ").strip()

        structured_blocks = {
            "Original Query": original_query,
            "Translated Concept": english_query,
            "Main Focal Centerpiece": focal_centerpiece or "Sacred cultural centerpiece",
            "Foreground Subject": english_query,
            "Spatial Composition": "Wide-angle dual-focal framing (Centerpiece background + Subject foreground)",
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
