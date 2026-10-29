import os
from typing import Dict
from huggingface_hub import InferenceClient
from src.multi_agent_pipeline.social_agent import get_llm, get_groq_llm, _call, _load_prompt

IMAGE_MODEL = "Qwen/Qwen-Image"

class SummarizingAgent:
    """
    Merges the final refined descriptions from all Social Agents into a
    single, coherent text-to-image generation prompt.
    """

    def __init__(self, llm=None, verbose: bool = False):
        self.llm = llm or get_groq_llm()
        self.final_caption = ""
        self.verbose = verbose

    def summarize(self, agent_descriptions: Dict[str, str]) -> str:
        """
        agent_descriptions: dict mapping agent name -> its final description,
        i.e. the return value of SocialAgentOrchestrator.run().
        """
        if self.verbose: print(f"[SummarizingAgent] summarize() start | {len(agent_descriptions)} descriptions")

        combined = "\n".join(f"- {desc}" for desc in agent_descriptions.values())

        print("< ---- Combined Context ---- >")
        if self.verbose: print(combined)

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
    token: str = None
):
    """
    Generate an image from a text prompt using FLUX.1-schnell via the
    Hugging Face Inference API, and save it to disk.
    """
   
    client = InferenceClient(
        provider="fal-ai",
        api_key=os.environ["HF_TOKEN"],
    )
    
    image = client.text_to_image(
        prompt=prompt,
        model=IMAGE_MODEL,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height
    )

    image.save(output_path)

    return output_path


if __name__ == "__main__":
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