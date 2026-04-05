"""
Master pipeline script to run the complete OpenAlex KG workflow.

Steps:
1. Fetch 5000 papers from OpenAlex
2. Extract PSP relations with Ollama
3. Build and visualize causal graph
4. Generate publication outputs (CSV, LaTeX, charts)

Usage:
    python run_full_pipeline.py                    # Run all steps
    python run_full_pipeline.py --step fetch       # Run only fetch step
    python run_full_pipeline.py --step extract     # Run only extraction
    python run_full_pipeline.py --step viz         # Run only visualization
    python run_full_pipeline.py --step publish     # Run only publication outputs
"""

import argparse
import sys
import subprocess
from pathlib import Path
import time


def run_step(step_name: str, script_name: str, description: str) -> bool:
    """
    Run a pipeline step and return success status.

    Args:
        step_name: Name of the step (for display)
        script_name: Python script to run
        description: Description of what this step does

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    print(f"{description}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            ['python', script_name],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )

        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} completed in {elapsed/60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {step_name} failed after {elapsed/60:.1f} minutes")
        print(f"  Error code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_name}")
        return False


def check_prerequisites() -> bool:
    """Check if all prerequisites are met."""
    print("Checking prerequisites...")

    # Check Ollama
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print("✗ Ollama is not running")
            print("  Start with: ollama serve")
            return False
        print("✓ Ollama is running")
    except FileNotFoundError:
        print("✗ Ollama is not installed")
        print("  Install: https://ollama.ai/")
        return False

    # Check if model is available
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if 'llama3.1:7b' not in result.stdout:
            print("✗ Model llama3.1:7b not found")
            print("  Download with: ollama pull llama3.1:7b")
            return False
        print("✓ Model llama3.1:7b is available")
    except FileNotFoundError:
        return False

    # Check required Python packages
    try:
        import requests
        import networkx
        import matplotlib
        print("✓ Required Python packages are installed")
    except ImportError as e:
        print(f"✗ Missing Python package: {e.name}")
        print("  Install with: pip install requests networkx matplotlib")
        return False

    print("✓ All prerequisites met\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete OpenAlex KG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py                 # Run complete pipeline
  python run_full_pipeline.py --step fetch    # Only fetch papers
  python run_full_pipeline.py --skip-check    # Skip prerequisite checks
        """
    )

    parser.add_argument(
        '--step',
        choices=['fetch', 'extract', 'viz', 'publish', 'all'],
        default='all',
        help='Which step to run (default: all)'
    )

    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='Skip prerequisite checks'
    )

    args = parser.parse_args()

    print(f"\n{'*'*60}")
    print("OpenAlex Knowledge Graph Pipeline")
    print(f"{'*'*60}\n")

    # Check prerequisites (unless skipped)
    if not args.skip_check:
        if not check_prerequisites():
            print("\n✗ Prerequisites not met. Fix issues above and try again.")
            print("  Or use --skip-check to bypass (not recommended)")
            return 1

    # Define pipeline steps
    steps = {
        'fetch': {
            'script': 'fetch_openalex_papers.py',
            'description': 'Fetch ~5000 papers from OpenAlex API'
        },
        'extract': {
            'script': 'extract_from_abstracts.py',
            'description': 'Extract PSP relations using Ollama (7B)'
        },
        'viz': {
            'script': 'visualize_kg.py',
            'description': 'Build and visualize causal graph (DAG)'
        },
        'publish': {
            'script': 'generate_publication_outputs.py',
            'description': 'Generate CSV, LaTeX, and chart outputs'
        }
    }

    # Determine which steps to run
    if args.step == 'all':
        steps_to_run = ['fetch', 'extract', 'viz', 'publish']
    else:
        steps_to_run = [args.step]

    # Run selected steps
    start_time = time.time()
    failed_steps = []

    for step_name in steps_to_run:
        step_info = steps[step_name]
        success = run_step(
            step_name.upper(),
            step_info['script'],
            step_info['description']
        )

        if not success:
            failed_steps.append(step_name)
            print(f"\n✗ Pipeline stopped due to failure in step: {step_name}")
            break

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")

    if failed_steps:
        print(f"✗ Failed steps: {', '.join(failed_steps)}")
        return 1
    else:
        print(f"✓ All steps completed successfully!")
        print(f"\nOutputs:")
        print(f"  - Knowledge Graph: outputs/kg_from_abstracts.json")
        print(f"  - Visualization: outputs/publication/causal_graph_openalex.png")
        print(f"  - CSV Tables: outputs/publication/*.csv")
        print(f"  - LaTeX Tables: outputs/publication/*.tex")
        print(f"  - Charts: outputs/publication/*.pdf/png")
        print(f"\nNext step: Integrate KG with ARIA reasoning engine")
        return 0


if __name__ == "__main__":
    sys.exit(main())
