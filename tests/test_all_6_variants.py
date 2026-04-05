"""
Comprehensive Test Suite for All 6 ARIA Variants

Tests all 6 variants on the same test cases for rigorous ablation study:
- BASELINE (Pure LLM)
- KG_ONLY (Pure Graph)
- NAIVE_KG (Simple KG + LLM)
- ARIA_CORE (3-Tier Reasoning)
- ARIA_SEARCH (+ Literature Search)
- ARIA_FULL (+ Chain-of-Thought)

Author: ARIA Team
Date: 2026-02-01
"""

import json
import time
from pathlib import Path
from typing import Dict, List
import traceback

# Import all variants
from src.variants.baseline_ollama import BaselineOllama
from src.variants.kg_only import KGOnly
from src.variants.naive_kg_ollama import NaiveKGOllama
from src.variants.aria_core_ollama import ARIACoreOllama
from src.variants.aria_search_ollama import ARIASearchOllama
from src.variants.aria_full_ollama import ARIAFullOllama


class VariantTester:
    """Test all variants on identical test cases."""

    def __init__(self, kg_file: str):
        """
        Initialize all variants.

        Args:
            kg_file: Path to knowledge graph JSON
        """
        self.kg_file = kg_file

        print("="*80)
        print("INITIALIZING ALL 6 ARIA VARIANTS")
        print("="*80)

        # Initialize all variants
        try:
            print("\n1. Initializing BASELINE...")
            self.baseline = BaselineOllama(model="qwen2:7b")
            print("   ✅ BASELINE ready")
        except Exception as e:
            print(f"   ❌ BASELINE failed: {e}")
            self.baseline = None

        try:
            print("\n2. Initializing KG_ONLY...")
            self.kg_only = KGOnly(kg_file=kg_file)
            print("   ✅ KG_ONLY ready")
        except Exception as e:
            print(f"   ❌ KG_ONLY failed: {e}")
            self.kg_only = None

        try:
            print("\n3. Initializing NAIVE_KG...")
            self.naive_kg = NaiveKGOllama(kg_file=kg_file, model="qwen2:7b")
            print("   ✅ NAIVE_KG ready")
        except Exception as e:
            print(f"   ❌ NAIVE_KG failed: {e}")
            self.naive_kg = None

        try:
            print("\n4. Initializing ARIA_CORE...")
            self.aria_core = ARIACoreOllama(kg_file=kg_file, model="qwen2:7b")
            print("   ✅ ARIA_CORE ready")
        except Exception as e:
            print(f"   ❌ ARIA_CORE failed: {e}")
            self.aria_core = None

        try:
            print("\n5. Initializing ARIA_SEARCH...")
            self.aria_search = ARIASearchOllama(kg_file=kg_file, model="qwen2:7b")
            print("   ✅ ARIA_SEARCH ready")
        except Exception as e:
            print(f"   ❌ ARIA_SEARCH failed: {e}")
            self.aria_search = None

        try:
            print("\n6. Initializing ARIA_FULL...")
            self.aria_full = ARIAFullOllama(kg_file=kg_file, model="qwen2:7b")
            print("   ✅ ARIA_FULL ready")
        except Exception as e:
            print(f"   ❌ ARIA_FULL failed: {e}")
            self.aria_full = None

        print("\n" + "="*80)
        print("ALL VARIANTS INITIALIZED")
        print("="*80)

    def test_forward(self, synthesis_inputs: Dict, test_name: str) -> Dict:
        """Test forward prediction on all variants."""
        print(f"\n{'='*80}")
        print(f"FORWARD TEST: {test_name}")
        print(f"{'='*80}")
        print(f"Inputs: {json.dumps(synthesis_inputs, indent=2)}")

        results = {}

        # Test BASELINE
        if self.baseline:
            print("\n[1/6] Testing BASELINE...")
            start_time = time.time()
            try:
                result = self.baseline.forward_prediction(synthesis_inputs)
                results['BASELINE'] = {
                    'success': True,
                    'latency': time.time() - start_time,
                    'confidence': result.get('confidence', 0),
                    'tier': result.get('tier', 3),
                    'reasoning_type': result.get('reasoning_type', 'unknown'),
                    'kg_paths': result.get('kg_paths', 0),
                    'result': result
                }
                print(f"   ✅ Success (latency={results['BASELINE']['latency']:.1f}s, confidence={results['BASELINE']['confidence']:.2f})")
            except Exception as e:
                results['BASELINE'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Failed: {e}")

        # Test KG_ONLY
        if self.kg_only:
            print("\n[2/6] Testing KG_ONLY...")
            start_time = time.time()
            try:
                result = self.kg_only.forward_prediction(synthesis_inputs)
                results['KG_ONLY'] = {
                    'success': True,
                    'latency': time.time() - start_time,
                    'confidence': result.get('confidence', 0),
                    'tier': 'N/A',
                    'reasoning_type': result.get('reasoning_type', 'unknown'),
                    'kg_paths': result.get('kg_paths', 0),
                    'result': result
                }
                print(f"   ✅ Success (latency={results['KG_ONLY']['latency']:.1f}s, confidence={results['KG_ONLY']['confidence']:.2f})")
            except Exception as e:
                results['KG_ONLY'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Failed: {e}")

        # Test NAIVE_KG
        if self.naive_kg:
            print("\n[3/6] Testing NAIVE_KG...")
            start_time = time.time()
            try:
                result = self.naive_kg.forward_prediction(synthesis_inputs)
                results['NAIVE_KG'] = {
                    'success': True,
                    'latency': time.time() - start_time,
                    'confidence': result.get('confidence', 0),
                    'tier': result.get('tier', 'N/A'),
                    'reasoning_type': result.get('reasoning_type', 'unknown'),
                    'kg_paths': result.get('kg_paths', 0),
                    'result': result
                }
                print(f"   ✅ Success (latency={results['NAIVE_KG']['latency']:.1f}s, confidence={results['NAIVE_KG']['confidence']:.2f})")
            except Exception as e:
                results['NAIVE_KG'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Failed: {e}")

        # Test ARIA_CORE
        if self.aria_core:
            print("\n[4/6] Testing ARIA_CORE...")
            start_time = time.time()
            try:
                result = self.aria_core.forward_prediction(synthesis_inputs)
                results['ARIA_CORE'] = {
                    'success': True,
                    'latency': time.time() - start_time,
                    'confidence': result.get('confidence', 0),
                    'tier': result.get('tier', 'unknown'),
                    'reasoning_type': result.get('reasoning_type', 'unknown'),
                    'kg_paths': result.get('kg_paths', 0),
                    'result': result
                }
                print(f"   ✅ Success (latency={results['ARIA_CORE']['latency']:.1f}s, confidence={results['ARIA_CORE']['confidence']:.2f})")
            except Exception as e:
                results['ARIA_CORE'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Failed: {e}")

        # Test ARIA_SEARCH
        if self.aria_search:
            print("\n[5/6] Testing ARIA_SEARCH...")
            start_time = time.time()
            try:
                result = self.aria_search.forward_prediction(synthesis_inputs)
                results['ARIA_SEARCH'] = {
                    'success': True,
                    'latency': time.time() - start_time,
                    'confidence': result.get('confidence', 0),
                    'tier': result.get('tier', 'unknown'),
                    'reasoning_type': result.get('reasoning_type', 'unknown'),
                    'kg_paths': result.get('kg_paths', 0),
                    'literature_papers': result.get('literature_papers', 0),
                    'result': result
                }
                print(f"   ✅ Success (latency={results['ARIA_SEARCH']['latency']:.1f}s, confidence={results['ARIA_SEARCH']['confidence']:.2f}, papers={results['ARIA_SEARCH']['literature_papers']})")
            except Exception as e:
                results['ARIA_SEARCH'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Failed: {e}")

        # Test ARIA_FULL
        if self.aria_full:
            print("\n[6/6] Testing ARIA_FULL...")
            start_time = time.time()
            try:
                result = self.aria_full.forward_prediction(synthesis_inputs)
                results['ARIA_FULL'] = {
                    'success': True,
                    'latency': time.time() - start_time,
                    'confidence': result.get('confidence', 0),
                    'tier': result.get('tier', 'unknown'),
                    'reasoning_type': result.get('reasoning_type', 'unknown'),
                    'kg_paths': result.get('kg_paths', 0),
                    'literature_papers': result.get('literature_papers', 0),
                    'cot_steps': len(result.get('chain_of_thought', {}).get('reasoning_steps', [])),
                    'result': result
                }
                print(f"   ✅ Success (latency={results['ARIA_FULL']['latency']:.1f}s, confidence={results['ARIA_FULL']['confidence']:.2f}, steps={results['ARIA_FULL']['cot_steps']})")
            except Exception as e:
                results['ARIA_FULL'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Failed: {e}")

        return results

    def test_inverse(self, desired_properties: Dict, test_name: str) -> Dict:
        """Test inverse design on all variants."""
        print(f"\n{'='*80}")
        print(f"INVERSE TEST: {test_name}")
        print(f"{'='*80}")
        print(f"Properties: {json.dumps(desired_properties, indent=2)}")

        results = {}

        # Test BASELINE
        if self.baseline:
            print("\n[1/6] Testing BASELINE...")
            start_time = time.time()
            try:
                result = self.baseline.inverse_design(desired_properties)
                results['BASELINE'] = {
                    'success': True,
                    'latency': time.time() - start_time,
                    'confidence': result.get('confidence', 0),
                    'tier': result.get('tier', 3),
                    'reasoning_type': result.get('reasoning_type', 'unknown'),
                    'kg_paths': result.get('kg_paths', 0),
                    'result': result
                }
                print(f"   ✅ Success (latency={results['BASELINE']['latency']:.1f}s, confidence={results['BASELINE']['confidence']:.2f})")
            except Exception as e:
                results['BASELINE'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Failed: {e}")

        # Test KG_ONLY
        if self.kg_only:
            print("\n[2/6] Testing KG_ONLY...")
            start_time = time.time()
            try:
                result = self.kg_only.inverse_design(desired_properties)
                results['KG_ONLY'] = {
                    'success': True,
                    'latency': time.time() - start_time,
                    'confidence': result.get('confidence', 0),
                    'tier': 'N/A',
                    'reasoning_type': result.get('reasoning_type', 'unknown'),
                    'kg_paths': result.get('kg_paths', 0),
                    'result': result
                }
                print(f"   ✅ Success (latency={results['KG_ONLY']['latency']:.1f}s, confidence={results['KG_ONLY']['confidence']:.2f})")
            except Exception as e:
                results['KG_ONLY'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Failed: {e}")

        # Test NAIVE_KG
        if self.naive_kg:
            print("\n[3/6] Testing NAIVE_KG...")
            start_time = time.time()
            try:
                result = self.naive_kg.inverse_design(desired_properties)
                results['NAIVE_KG'] = {
                    'success': True,
                    'latency': time.time() - start_time,
                    'confidence': result.get('confidence', 0),
                    'tier': result.get('tier', 'N/A'),
                    'reasoning_type': result.get('reasoning_type', 'unknown'),
                    'kg_paths': result.get('kg_paths', 0),
                    'result': result
                }
                print(f"   ✅ Success (latency={results['NAIVE_KG']['latency']:.1f}s, confidence={results['NAIVE_KG']['confidence']:.2f})")
            except Exception as e:
                results['NAIVE_KG'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Failed: {e}")

        # Test ARIA_CORE
        if self.aria_core:
            print("\n[4/6] Testing ARIA_CORE...")
            start_time = time.time()
            try:
                result = self.aria_core.inverse_design(desired_properties)
                results['ARIA_CORE'] = {
                    'success': True,
                    'latency': time.time() - start_time,
                    'confidence': result.get('confidence', 0),
                    'tier': result.get('tier', 'unknown'),
                    'reasoning_type': result.get('reasoning_type', 'unknown'),
                    'kg_paths': result.get('kg_paths', 0),
                    'result': result
                }
                print(f"   ✅ Success (latency={results['ARIA_CORE']['latency']:.1f}s, confidence={results['ARIA_CORE']['confidence']:.2f})")
            except Exception as e:
                results['ARIA_CORE'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Failed: {e}")

        # Test ARIA_SEARCH
        if self.aria_search:
            print("\n[5/6] Testing ARIA_SEARCH...")
            start_time = time.time()
            try:
                result = self.aria_search.inverse_design(desired_properties)
                results['ARIA_SEARCH'] = {
                    'success': True,
                    'latency': time.time() - start_time,
                    'confidence': result.get('confidence', 0),
                    'tier': result.get('tier', 'unknown'),
                    'reasoning_type': result.get('reasoning_type', 'unknown'),
                    'kg_paths': result.get('kg_paths', 0),
                    'literature_papers': result.get('literature_papers', 0),
                    'result': result
                }
                print(f"   ✅ Success (latency={results['ARIA_SEARCH']['latency']:.1f}s, confidence={results['ARIA_SEARCH']['confidence']:.2f}, papers={results['ARIA_SEARCH']['literature_papers']})")
            except Exception as e:
                results['ARIA_SEARCH'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Failed: {e}")

        # Test ARIA_FULL
        if self.aria_full:
            print("\n[6/6] Testing ARIA_FULL...")
            start_time = time.time()
            try:
                result = self.aria_full.inverse_design(desired_properties)
                results['ARIA_FULL'] = {
                    'success': True,
                    'latency': time.time() - start_time,
                    'confidence': result.get('confidence', 0),
                    'tier': result.get('tier', 'unknown'),
                    'reasoning_type': result.get('reasoning_type', 'unknown'),
                    'kg_paths': result.get('kg_paths', 0),
                    'literature_papers': result.get('literature_papers', 0),
                    'cot_steps': len(result.get('chain_of_thought', {}).get('reasoning_steps', [])),
                    'result': result
                }
                print(f"   ✅ Success (latency={results['ARIA_FULL']['latency']:.1f}s, confidence={results['ARIA_FULL']['confidence']:.2f}, steps={results['ARIA_FULL']['cot_steps']})")
            except Exception as e:
                results['ARIA_FULL'] = {'success': False, 'error': str(e)}
                print(f"   ❌ Failed: {e}")

        return results

    def run_all_tests(self) -> Dict:
        """Run complete test suite."""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE 6-VARIANT ABLATION STUDY")
        print("="*80)

        all_results = {
            'forward_tests': {},
            'inverse_tests': {}
        }

        # Define test cases
        forward_tests = [
            {
                'name': 'CVD_MoS2_Nb',
                'inputs': {
                    'method': 'CVD',
                    'host_material': 'MoS2',
                    'dopant': 'Nb'
                }
            },
            {
                'name': 'Oxidation_High_Temp',
                'inputs': {
                    'method': 'high temperature oxidation',
                    'temperature': '200°C'
                }
            },
            {
                'name': 'Phosphorus_Doping',
                'inputs': {
                    'dopant': 'phosphorus',
                    'method': 'doping'
                }
            }
        ]

        inverse_tests = [
            {
                'name': 'N_type_High_Mobility',
                'properties': {
                    'carrier_type': 'n-type',
                    'mobility': 'high'
                }
            },
            {
                'name': 'P_type_Doping',
                'properties': {
                    'conductivity': 'p-type',
                    'doping': 'p-type'
                }
            }
        ]

        # Run forward tests
        for test_case in forward_tests:
            results = self.test_forward(test_case['inputs'], test_case['name'])
            all_results['forward_tests'][test_case['name']] = results

        # Run inverse tests
        for test_case in inverse_tests:
            results = self.test_inverse(test_case['properties'], test_case['name'])
            all_results['inverse_tests'][test_case['name']] = results

        return all_results

    def analyze_results(self, all_results: Dict):
        """Analyze and summarize results."""
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS SUMMARY")
        print("="*80)

        variants = ['BASELINE', 'KG_ONLY', 'NAIVE_KG', 'ARIA_CORE', 'ARIA_SEARCH', 'ARIA_FULL']

        # Collect metrics
        metrics = {v: {'latency': [], 'confidence': [], 'success': 0, 'total': 0} for v in variants}

        for test_type in ['forward_tests', 'inverse_tests']:
            for test_name, results in all_results[test_type].items():
                for variant in variants:
                    if variant in results:
                        metrics[variant]['total'] += 1
                        if results[variant].get('success'):
                            metrics[variant]['success'] += 1
                            metrics[variant]['latency'].append(results[variant].get('latency', 0))
                            metrics[variant]['confidence'].append(results[variant].get('confidence', 0))

        # Print summary table
        print("\n| Variant | Success Rate | Avg Latency | Avg Confidence |")
        print("|---------|--------------|-------------|----------------|")

        for variant in variants:
            m = metrics[variant]
            success_rate = f"{m['success']}/{m['total']}" if m['total'] > 0 else "N/A"
            avg_latency = f"{np.mean(m['latency']):.1f}s" if m['latency'] else "N/A"
            avg_confidence = f"{np.mean(m['confidence']):.2f}" if m['confidence'] else "N/A"
            print(f"| **{variant}** | {success_rate} | {avg_latency} | {avg_confidence} |")

        print("\n" + "="*80)
        print("COMPONENT CONTRIBUTION ANALYSIS")
        print("="*80)

        if metrics['NAIVE_KG']['confidence'] and metrics['BASELINE']['confidence']:
            kg_contribution = np.mean(metrics['NAIVE_KG']['confidence']) - np.mean(metrics['BASELINE']['confidence'])
            print(f"\nKG Contribution (NAIVE_KG - BASELINE): {kg_contribution:+.3f}")

        if metrics['ARIA_CORE']['confidence'] and metrics['NAIVE_KG']['confidence']:
            tier_contribution = np.mean(metrics['ARIA_CORE']['confidence']) - np.mean(metrics['NAIVE_KG']['confidence'])
            print(f"Tier Reasoning Contribution (ARIA_CORE - NAIVE_KG): {tier_contribution:+.3f}")

        if metrics['ARIA_SEARCH']['confidence'] and metrics['ARIA_CORE']['confidence']:
            search_contribution = np.mean(metrics['ARIA_SEARCH']['confidence']) - np.mean(metrics['ARIA_CORE']['confidence'])
            print(f"Literature Search Contribution (ARIA_SEARCH - ARIA_CORE): {search_contribution:+.3f}")

        if metrics['ARIA_FULL']['confidence'] and metrics['ARIA_SEARCH']['confidence']:
            cot_contribution = np.mean(metrics['ARIA_FULL']['confidence']) - np.mean(metrics['ARIA_SEARCH']['confidence'])
            print(f"Chain-of-Thought Contribution (ARIA_FULL - ARIA_SEARCH): {cot_contribution:+.3f}")


if __name__ == '__main__':
    import numpy as np

    # Use production KG
    kg_file = "../data/KG/outputs/combined_doping_data.json"

    # Initialize tester
    tester = VariantTester(kg_file=kg_file)

    # Run all tests
    all_results = tester.run_all_tests()

    # Save detailed results
    output_file = "test_results_6_variants.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ Detailed results saved to: {output_file}")

    # Analyze and print summary
    tester.analyze_results(all_results)

    print("\n" + "="*80)
    print("ALL 6 VARIANTS TESTED SUCCESSFULLY!")
    print("="*80)
