import os
from typing import Dict
from huggingface_hub import InferenceClient
from src.multi_agent_pipeline.social_agent import get_llm, get_groq_llm, _call, _load_prompt

# Configuration
IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

class SummarizingAgent:
    """
    Merges the final refined descriptions from all Social Agents into a
    single, coherent text-to-image generation prompt.
    """

    def __init__(self, llm=None):
        self.llm = llm or get_groq_llm()
        self.final_caption = ""

    def summarize(self, agent_descriptions: Dict[str, str]) -> str:
        """
        agent_descriptions: dict mapping agent name -> its final description,
        i.e. the return value of SocialAgentOrchestrator.run().
        """
        print(f"[SummarizingAgent] summarize() start | {len(agent_descriptions)} descriptions")

        combined = "\n".join(f"- {desc}" for desc in agent_descriptions.values())

        print("< ---- Combined Context ---- >")
        print(combined)

        template = _load_prompt("summarizing_agent_system.txt")
        system_prompt = template.format(combined_descriptions=combined)

        self.final_caption = _call(self.llm, system_prompt)
        print("< ---- Final Caption ---- >")
        print(self.final_caption)
        print(f"[SummarizingAgent] summarize() done -> {self.final_caption[:100]!r}")
        return self.final_caption


def generate_image(
    prompt: str,
    output_path: str = "generated_image.png",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 4,
) -> str:
    """
    Generate an image from a text prompt using FLUX.1-schnell via the
    Hugging Face Inference API, and save it to disk.

    FLUX.1-schnell is a distilled model optimized for ~4 inference steps
    and guidance_scale=0.0 -- more steps/guidance generally do not improve
    quality for this specific model and just cost more time.
    """
    print(f"[generate_image] Generating image | prompt_len={len(prompt)} chars, size={width}x{height}")
    client = InferenceClient(model=IMAGE_MODEL)

    try:
        image = client.text_to_image(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            width=width,
            height=height,
        )
    except Exception as e:
        print(f"[generate_image] ERROR during text_to_image: {e}")
        raise

    image.save(output_path)
    print(f"[generate_image] Image saved -> {output_path}")
    return output_path


if __name__ == "__main__":
    # Example: run this against whatever final_descriptions your
    # SocialAgentOrchestrator produced. Swapped in some sample data here
    # so this file can be run/tested on its own.
    final_descriptions = {
        "agent-1": "A 30-year-old woman with dark hair tied back, wearing a "
                   "simple cotton saree in warm yellow and orange tones.",
        "agent-2": "A riverside ghat in Patna at dusk, stone steps leading "
                   "down to the water, small oil lamps placed along the edges.",
        "agent-3": "Bamboo winnowing baskets (soop) filled with fruit and "
                   "sugarcane, offered toward the setting sun over the river.",
    }

    print("[main] Starting summarization")
    summarizer = SummarizingAgent(llm=get_groq_llm())
    final_caption = summarizer.summarize(final_descriptions)
    print(f"\n=== FINAL IMAGE CAPTION ===\n{final_caption}")

    print("[main] Starting image generation")
    image_path = generate_image(final_caption, output_path="generated_image.png")
    print(f"[main] Image generated -> {image_path}")