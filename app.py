import gradio as gr
from src.pipeline import CulturalPromptPipeline

print("Loading Cultural Prompt Pipeline...")
pipeline = CulturalPromptPipeline()


def generate_prompt_ui(query_text: str):
    if not query_text or not query_text.strip():
        return "Please enter a query in any Indian language.", "", "", ""
    
    result = pipeline.run(query_text, return_details=True)
    
    inter = result.get("intermediate", {})
    
    sources_text = "\n\n".join([
        f"--- Source: {src} ---\n{txt}"
        for src, txt in zip(inter.get("retrieved_context_sources", []), inter.get("retrieved_raw_texts", []))
    ])
    
    filter_lines = []
    for cat, details in inter.get("filtered_categories", {}).items():
        if details:
            filter_lines.append(f"### ✦ {cat.replace('_', ' ').title()}\n" + "\n".join([f"• {d}" for d in details]))
    filtered_text = "\n\n".join(filter_lines) if filter_lines else "No visual details extracted."
    
    # Structured prompt summary
    full_prompt = result.get("structured_prompt", "")
    negative_prompt = result.get("negative_prompt", "")
    
    return full_prompt, negative_prompt, filtered_text, sources_text

SAMPLE_QUERIES = [
    ["छठ पूजा अर्घ्य देती महिला"],
    ["দুর্গাপূজায় ধুনুচি নাচ"],
    ["பொங்கல் திருநாள் கொண்டாட்டம்"],
    ["సంక్రాంతి భోగి మంటలు"],
    ["गुढीपाडवा मिरवणूक"],
    ["Kathakali dancer expressing Navarasa"],
    ["Warli tribal celebration with Tarpa dance"]
]

custom_css = """
.container { max-width: 1100px; margin: auto; padding: 20px; }
.header-box { text-align: center; margin-bottom: 25px; padding: 25px; background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%); border-radius: 12px; border: 1px solid #3d3d5c; color: #ffffff; }
.header-title { font-size: 2.2rem; font-weight: 700; color: #f39c12; margin-bottom: 8px; }
.header-sub { font-size: 1.05rem; color: #d1d5db; }
.output-box textarea { font-family: 'Fira Code', 'Courier New', monospace; font-size: 0.95rem; line-height: 1.5; }
"""

with gr.Blocks(title="Multilingual Cultural Prompt-Generation Pipeline") as demo:
    gr.HTML("""
    <div class="header-box">
        <h1 class="header-title">🪔 Multilingual Cultural Prompt Pipeline</h1>
        <p class="header-sub">User Query ➔ IndicBERT ➔ Cultural Web Retrieval ➔ Context Filtering ➔ Structured SD Prompt</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=5):
            query_input = gr.Textbox(
                label="Enter Query (Any Indian Language)",
                placeholder="e.g., छठ पूजा अर्घ्य देती महिला, দুর্গাপূজায় ধুনুচি নাচ, பொங்கல் திருநாள் கொண்டாட்டம்...",
                lines=2
            )
            submit_btn = gr.Button("✨ Generate Structured Prompt", variant="primary", size="lg")
            
            gr.Examples(
                examples=SAMPLE_QUERIES,
                inputs=query_input,
                label="Sample Multilingual Test Queries"
            )
            
        with gr.Column(scale=7):
            final_prompt_out = gr.Textbox(
                label="🎯 Structured Image-Generation Prompt (Stable Diffusion / FLUX ready)",
                lines=5,
                buttons=["copy"],
                elem_classes=["output-box"]
            )
            negative_prompt_out = gr.Textbox(
                label="🚫 Recommended Negative Prompt",
                lines=3,
                buttons=["copy"]
            )

    with gr.Accordion("🔍 Pipeline Inspection: IndicBERT Context Filtering & Web Retrieval", open=False):
        with gr.Row():
            with gr.Column():
                filtered_details_out = gr.Markdown(label="Filtered Cultural Visual Details (IndicBERT Token-Efficient Filtering)")
            with gr.Column():
                raw_sources_out = gr.Textbox(label="Retrieved Raw Web/Wikipedia Context", lines=10)

    submit_btn.click(
        fn=generate_prompt_ui,
        inputs=[query_input],
        outputs=[final_prompt_out, negative_prompt_out, filtered_details_out, raw_sources_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
