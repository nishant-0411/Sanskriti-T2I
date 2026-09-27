import argparse
import sys
import json
from src.pipeline import CulturalPromptPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual Cultural Prompt-Generation Pipeline (User Query -> IndicBERT -> Cultural Web Retrieval -> Context Filtering -> Structured Prompt)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="छठ पूजा अर्घ्य देती महिला",
        help="Input query in any Indian language or English (e.g., 'छठ पूजा अर्घ्य देती महिला', 'দুর্গাপূজায় ধুনুচি নাচ', 'Kathakali dancer')"
    )
    parser.add_argument(
        "--prompt_only",
        action="store_true",
        default=True,
        help="Output ONLY the final structured image-generation prompt string."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full intermediate pipeline execution details (IndicBERT embeddings, retrieval, filtered categories)."
    )

    args = parser.parse_args()

    # Initialize and run pipeline
    pipeline = CulturalPromptPipeline()
    result = pipeline.run(args.query, return_details=args.verbose)

    if args.verbose:
        print("\n" + "="*80)
        print("MULTILINGUAL CULTURAL PROMPT-GENERATION PIPELINE EXECUTION SUMMARY")
        print("="*80)
        print(f"User Query: {args.query}")
        print(f"IndicBERT Model: {result['intermediate']['indic_bert_model']}")
        print(f"English Concept: {result['intermediate']['english_query']}")
        print("\n[Retrieved Context Sources]:")
        for src in result['intermediate']['retrieved_context_sources']:
            print(f"  - {src}")
            
        print("\n[Filtered Visual Categories via IndicBERT]:")
        for cat, details in result['intermediate']['filtered_categories'].items():
            if details:
                print(f"  * {cat.upper()}:")
                for d in details:
                    print(f"      - {d}")
                    
        print("\n[Structured Prompt Blocks]:")
        for k, v in result['prompt_blocks'].items():
            print(f"  {k}: {v}")
            
        print("\n[Negative Prompt]:")
        print(result['negative_prompt'])
        print("\n" + "="*80)
        print("FINAL STRUCTURED STABLE-DIFFUSION PROMPT:")
        print("="*80)
        print(result['structured_prompt'])
        print("="*80 + "\n")
    else:
        # Output ONLY the final structured prompt as strictly required
        print(result['structured_prompt'])

if __name__ == "__main__":
    main()
