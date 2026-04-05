"""
Comprehensive Test Suite for All ARIA Variants

Tests all 4 implemented variants on the same queries to enable rigorous comparison.
Uses production KG (combined_doping_data.json) with 777 nodes and 409 edges.

Author: ARIA Team
Date: 2026-02-01
"""

import json
import time
from typing import Dict, List, Any
from pathlib import Path

# Import all variants
from src.variants.baseline_ollama import BaselineOllama
from src.variants.kg_only import KGOnly
from src.variants.naive_kg_ollama import NaiveKGOllama
from src.variants.aria_core_ollama import ARIACoreOllama


class VariantTester:
    """Test all ARIA variants on common queries."""

    def __init__(self, kg_file: str):
        """
        Initialize all variants with production KG.

        Args:
            kg_file: Path to production KG
        """
        self.kg_file = kg_file

        print("Initializing all variants...")
        print(f"KG: {kg_file}")

        # Initialize variants
        self.baseline = BaselineOllama(model="qwen2:7b")
        self.kg_only = KGOnly(kg_file=kg_file)
        self.naive_kg = NaiveKGOllama(kg_file=kg_file, model="qwen2:7b")
        self.aria_core = ARIACoreOllama(kg_file=kg_file, model="qwen2:7b")

        print("✅ All variants initialized\n")

    def get_test_cases(self) -> List[Dict[str, Any]]:
        """Get test cases for evaluation."""
        return [
            {
                'id': 'forward_1',
                'type': 'forward',
                'name': 'CVD MoS2 Nb doping',
                'input': {
                    "method": "CVD",
                    "temperature_c": 750,
                    "material": "MoS2",
                    "dopant": "Nb",
                    "time_hours": 1
                },
                'expected_coverage': 'high'  # Should match KG
            },
            {
                'id': 'forward_2',
                'type': 'forward',
                'name': 'High temperature oxidation',
                'input': {
                    "method": "Oxidation",
                    "temperature_c": 800,
                    "atmosphere": "oxygen"
                },
                'expected_coverage': 'medium'
            },
            {
                'id': 'forward_3',
                'type': 'forward',
                'name': 'Phosphorus doping',
                'input': {
                    "dopant": "phosphorus",
                    "concentration": "high",
                    "method": "ion implantation"
                },
                'expected_coverage': 'high'  # KG has phosphorus
            },
            {
                'id': 'inverse_1',
                'type': 'inverse',
                'name': 'N-type high mobility',
                'input': {
                    "carrier_type": "n-type",
                    "mobility": "high (>50 cm2/V·s)",
                    "material": "2D material"
                },
                'expected_coverage': 'low'  # Inverse design harder
            },
            {
                'id': 'inverse_2',
                'type': 'inverse',
                'name': 'P-type doping',
                'input': {
                    "carrier_type": "p-type",
                    "doping_type": "controllable"
                },
                'expected_coverage': 'medium'  # KG has p-type
            }
        ]

    def run_variant(
        self,
        variant_name: str,
        variant,
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run single test case on single variant."""
        start_time = time.time()

        try:
            if test_case['type'] == 'forward':
                result = variant.forward_prediction(test_case['input'])
            else:  # inverse
                result = variant.inverse_design(test_case['input'])

            latency = time.time() - start_time

            return {
                'variant': variant_name,
                'test_id': test_case['id'],
                'test_name': test_case['name'],
                'status': 'success',
                'result': result,
                'latency': latency
            }

        except Exception as e:
            latency = time.time() - start_time
            return {
                'variant': variant_name,
                'test_id': test_case['id'],
                'test_name': test_case['name'],
                'status': 'error',
                'error': str(e),
                'latency': latency
            }

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all test cases on all variants."""
        test_cases = self.get_test_cases()
        results = []

        variants = [
            ('BASELINE', self.baseline),
            ('KG_ONLY', self.kg_only),
            ('NAIVE_KG', self.naive_kg),
            ('ARIA_CORE', self.aria_core)
        ]

        total_tests = len(test_cases) * len(variants)
        current_test = 0

        print(f"Running {total_tests} tests ({len(test_cases)} cases × {len(variants)} variants)\n")
        print("=" * 80)

        for test_case in test_cases:
            print(f"\n📝 Test Case: {test_case['name']} ({test_case['type']})")
            print(f"   Input: {json.dumps(test_case['input'], indent=2)}")
            print("-" * 80)

            for variant_name, variant in variants:
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] Running {variant_name}...")

                result = self.run_variant(variant_name, variant, test_case)
                results.append(result)

                # Print summary
                if result['status'] == 'success':
                    res = result['result']
                    conf = res.get('confidence', 'N/A')
                    tier = res.get('tier', 'N/A')
                    reasoning_type = res.get('reasoning_type', 'N/A')
                    kg_paths = res.get('kg_paths_used', res.get('paths_found', 'N/A'))

                    print(f"   ✅ Success (latency: {result['latency']:.1f}s)")
                    print(f"      Confidence: {conf}")
                    print(f"      Tier: {tier}")
                    print(f"      Reasoning: {reasoning_type}")
                    print(f"      KG paths: {kg_paths}")
                else:
                    print(f"   ❌ Error: {result['error']}")

            print("\n" + "=" * 80)

        return results

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and summarize results."""
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)

        # Group by variant
        by_variant = {}
        for r in results:
            variant = r['variant']
            if variant not in by_variant:
                by_variant[variant] = []
            by_variant[variant].append(r)

        summary = {}

        for variant_name, variant_results in by_variant.items():
            successes = [r for r in variant_results if r['status'] == 'success']
            errors = [r for r in variant_results if r['status'] == 'error']

            # Calculate metrics
            success_rate = len(successes) / len(variant_results) if variant_results else 0
            avg_latency = sum(r['latency'] for r in successes) / len(successes) if successes else 0
            avg_confidence = sum(r['result'].get('confidence', 0) for r in successes) / len(successes) if successes else 0

            # Tier usage (ARIA_CORE only)
            tier_usage = {}
            if variant_name == 'ARIA_CORE':
                for r in successes:
                    tier = r['result'].get('tier', 'unknown')
                    tier_usage[f'Tier {tier}'] = tier_usage.get(f'Tier {tier}', 0) + 1

            # KG usage
            kg_paths_used = [r['result'].get('kg_paths_used', r['result'].get('paths_found', 0)) for r in successes]
            avg_kg_paths = sum(kg_paths_used) / len(kg_paths_used) if kg_paths_used else 0
            queries_with_kg = sum(1 for p in kg_paths_used if p > 0)
            kg_usage_rate = queries_with_kg / len(kg_paths_used) if kg_paths_used else 0

            summary[variant_name] = {
                'total_tests': len(variant_results),
                'successes': len(successes),
                'errors': len(errors),
                'success_rate': success_rate,
                'avg_latency': avg_latency,
                'avg_confidence': avg_confidence,
                'avg_kg_paths': avg_kg_paths,
                'kg_usage_rate': kg_usage_rate,
                'tier_usage': tier_usage if tier_usage else None
            }

        # Print summary
        for variant_name, stats in summary.items():
            print(f"\n{variant_name}:")
            print(f"  Success Rate:      {stats['success_rate']:.1%} ({stats['successes']}/{stats['total_tests']})")
            print(f"  Avg Latency:       {stats['avg_latency']:.1f}s")
            print(f"  Avg Confidence:    {stats['avg_confidence']:.2f}")
            print(f"  Avg KG Paths:      {stats['avg_kg_paths']:.1f}")
            print(f"  KG Usage Rate:     {stats['kg_usage_rate']:.1%}")
            if stats['tier_usage']:
                print(f"  Tier Usage:        {stats['tier_usage']}")

        return summary

    def save_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any], output_file: str = "test_results.json"):
        """Save results to file."""
        output = {
            'timestamp': '2026-02-01',
            'kg_file': str(self.kg_file),
            'results': results,
            'summary': summary
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n✅ Results saved to: {output_file}")


def main():
    """Run comprehensive variant testing."""
    print("\n" + "=" * 80)
    print("ARIA VARIANT COMPREHENSIVE TESTING")
    print("=" * 80)

    kg_file = "../data/KG/outputs/combined_doping_data.json"

    try:
        tester = VariantTester(kg_file)
        results = tester.run_all_tests()
        summary = tester.analyze_results(results)
        tester.save_results(results, summary)

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETE!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
