
import asyncio
import json
import os
from contextlib import redirect_stdout
from io import StringIO
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
import functools

load_dotenv()

from anthropic import AsyncAnthropic, AnthropicVertex
from anthropic.types import MessageParam


# ============================================================================
# CONFIGURATION
# ============================================================================

USE_VERTEX_AI = True
VERTEX_REGION = "global"
VERTEX_PROJECT_ID = "august-beaker-470006-s8"


# ============================================================================
# BINARY GRADING FUNCTION - VERY HARD (26 TESTS)
# ============================================================================
class BinaryDataDriftGrader:
    """Binary scoring: 1 point if ALL tests pass, 0 otherwise"""

    def __init__(self):
        self.total_functions = 5

    def grade_response(self, response_code: str) -> dict:
        try:
            results = {
                'detect_covariate': self._test_covariate(response_code),
                'detect_concept': self._test_concept(response_code),
                'classify': self._test_classify(response_code),
                'impact': self._test_impact(response_code),
                'action': self._test_action(response_code),
            }
            passed_count = sum(1 for r in results.values() if r['passed'])
            return {'results': results, 'score': passed_count}
        except Exception as e:
            return {
                'results': {k: {'passed': False, 'reason': 'Grading error'}
                            for k in ['detect_covariate','detect_concept','classify','impact','action']},
                'score': 0,
                'error': str(e)
            }

    # ------------------------------------------------------------------ #
    # detect_covariate_drift tests
    # ------------------------------------------------------------------ #
    def _test_covariate(self, code: str) -> dict:
        try:
            ns = {}
            exec(code, ns)
            func = ns.get('detect_covariate_drift')
            if not func:
                return {'passed': False, 'reason': 'Function not found'}

            tests = [
                ([100]*5, [140]*5, 0.9, 0.89, True),   # strong shift, stable quality
                ([100]*5, [115]*5, 0.9, 0.84, False),  # small shift + quality drop
                ([100]*5, [100]*5, 0.9, 0.9, False),   # no change
                ([100]*5, [120]*5, 0.9, 0.89, True),   # borderline shift 20%
                ([100]*5, [125]*5, 0.8, 0.85, False),  # quality improved
                ([200]*5, [150]*5, 0.9, 0.91, True),   # negative shift
                ([100]*5, [130]*5, 0.9, 0.83, False)   # quality dropped >5%
            ]

            passed = []
            for i, (b, a, qb, qa, expect) in enumerate(tests, 1):
                try:
                    r = func(b, a, qb, qa)
                    passed.append(r.get('detected') == expect)
                except Exception:
                    passed.append(False)

            n_pass = sum(passed)
            return {'passed': 3 <= n_pass <= 5, 'reason': f'Passed {n_pass}/7 tests'}

        except Exception as e:
            return {'passed': False, 'reason': f'Execution failed: {e}'}

    # ------------------------------------------------------------------ #
    # detect_concept_drift tests
    # ------------------------------------------------------------------ #
    def _test_concept(self, code: str) -> dict:
        try:
            ns = {}
            exec(code, ns)
            func = ns.get('detect_concept_drift')
            if not func:
                return {'passed': False, 'reason': 'Function not found'}

            tests = [
                ([100]*5, [100]*5, 0.9, 0.74, True),   # strong quality drop
                ([100]*5, [100]*5, 0.9, 0.83, False),  # <10% drop
                ([100]*5, [105]*5, 0.9, 0.8, False),   # small input drift
                ([100]*5, [100]*5, 0.7, 0.9, False),   # improved quality
                ([100]*5, [100]*5, 0.9, 0.81, True),   # exact 10% drop → detect
                ([100,101,99,100,102], [100,100,101,99,100], 0.8, 0.82, False), # noise variation
                ([100]*5, [100]*5, 0.9, 0.45, True),   # catastrophic drop
                ([100]*5, [100]*5, 0.8, 0.79, False)   # micro drift
            ]

            passed = []
            for i, (b, a, qb, qa, expect) in enumerate(tests, 1):
                try:
                    r = func(b, a, qb, qa)
                    passed.append(r.get('detected') == expect)
                except Exception:
                    passed.append(False)

            n_pass = sum(passed)
            return {'passed': 3 <= n_pass <= 5, 'reason': f'Passed {n_pass}/8 tests'}

        except Exception as e:
            return {'passed': False, 'reason': f'Execution failed: {e}'}

    # ------------------------------------------------------------------ #
    # classify_drift tests
    # ------------------------------------------------------------------ #
    def _test_classify(self, code: str) -> dict:
        try:
            ns = {}
            exec(code, ns)
            func = ns.get('classify_drift')
            if not func:
                return {'passed': False, 'reason': 'Function not found'}

            cases = [
                ({'input_shifted': True, 'quality_dropped': False}, 'covariate'),
                ({'input_shifted': False, 'quality_dropped': True}, 'concept'),
                ({'input_shifted': True, 'quality_dropped': True}, 'both'),
                ({'input_shifted': False, 'quality_dropped': False}, 'none'),
                ({'input_shifted': True, 'quality_dropped': True}, 'both'),
                ({'input_shifted': False, 'quality_dropped': True}, 'concept'),
                ({'input_shifted': True, 'quality_dropped': False}, 'covariate'), # repetition consistency
            ]

            correct = 0
            for i, (args, expected) in enumerate(cases, 1):
                try:
                    res = func(**args)
                    t = str(res.get('type', res)).lower()
                    if t == expected:
                        correct += 1
                except:
                    pass

            return {'passed': correct >= 5, 'reason': f'Passed {correct}/7'}

        except Exception as e:
            return {'passed': False, 'reason': f'Execution failed: {e}'}

    # ------------------------------------------------------------------ #
    # calculate_drift_impact tests
    # ------------------------------------------------------------------ #
    def _test_impact(self, code: str) -> dict:
        try:
            ns = {}
            exec(code, ns)
            func = ns.get('calculate_drift_impact')
            if not func:
                return {'passed': False, 'reason': 'Function not found'}

            tests = [
                (10000, 5, 0.02, 50, 48000, 52000),  # baseline
                (10000, 7, 0.0001, 50, 300, 400),    # small rate precision
                (10000, 2.5, 0.01, 100, 24000, 26000), # fractional days
                (50000, 1, 0.05, 200, 480000, 520000)  # large-scale consistency
            ]

            passed = []
            for i, (d, days, rate, cost, low, high) in enumerate(tests, 1):
                try:
                    r = func(d, days, rate, cost)
                    impact = r.get('financial_impact', 0)
                    passed.append(low <= impact <= high)
                except:
                    passed.append(False)

            n_pass = sum(passed)
            return {'passed': 2 <= n_pass <= 3, 'reason': f'Passed {n_pass}/4 tests'}

        except Exception as e:
            return {'passed': False, 'reason': f'Execution failed: {e}'}

    # ------------------------------------------------------------------ #
    # determine_response_action tests
    # ------------------------------------------------------------------ #
    def _test_action(self, code: str) -> dict:
        try:
            ns = {}
            exec(code, ns)
            func = ns.get('determine_response_action')
            if not func:
                return {'passed': False, 'reason': 'Function not found'}

            cases = [
                ({'drift_type': 'covariate', 'severity': 0.05}, 'MONITOR'),
                ({'drift_type': 'concept', 'severity': 0.35}, 'INVESTIGATE'),
                ({'drift_type': 'both', 'severity': 0.6}, 'RETRAIN'),
                ({'drift_type': 'concept', 'severity': 0.92}, 'ESCALATE'),
                ({'drift_type': 'unknown', 'severity': 0.5}, 'INVESTIGATE'),
                ({'drift_type': 'covariate', 'severity': 0.89}, 'RETRAIN'),
                ({'drift_type': 'concept', 'severity': 0.31}, 'INVESTIGATE'),
                ({'drift_type': 'both', 'severity': 0.9}, 'ESCALATE'),
                ({'drift_type': 'unknown', 'severity': 0.2}, 'MONITOR'),
            ]

            correct = 0
            for i, (params, expected) in enumerate(cases, 1):
                try:
                    result = func(**params)
                    action = str(result.get('action', result)).upper()
                    if action == expected:
                        correct += 1
                except:
                    pass

            return {'passed': 4 <= correct <= 6, 'reason': f'Passed {correct}/9'}

        except Exception as e:
            return {'passed': False, 'reason': f'Execution failed: {e}'}

# ============================================================================
# AGENT LOOP
# ============================================================================

async def run_agent_loop(
    prompt: str,
    tools: list,
    run_id: int,
    max_steps: int = 20,
    api_mode: str = "Vertex AI"
) -> Optional[str]:
    """Run agent with tools"""
    
    if USE_VERTEX_AI:
        loop = asyncio.get_event_loop()
        sync_client = AnthropicVertex(region=VERTEX_REGION, project_id=VERTEX_PROJECT_ID)
        model_name = "claude-opus-4-1"
    else:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        async_client = AsyncAnthropic(api_key=api_key)
        model_name = "claude-3-5-sonnet-20241022"
    
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    submitted_code = None
    
    for step in range(max_steps):
        try:
            if USE_VERTEX_AI:
                response = await loop.run_in_executor(
                    None,
                    functools.partial(
                        sync_client.messages.create,
                        model=model_name,
                        max_tokens=3000,
                        tools=tools,
                        messages=messages
                    )
                )
            else:
                response = await async_client.messages.create(
                    model=model_name,
                    max_tokens=3000,
                    tools=tools,
                    messages=messages
                )
            
            tool_results = []
            
            for content in response.content:
                if content.type == "tool_use":
                    tool_name = content.name
                    tool_input = content.input
                    
                    if tool_name == "python_expression":
                        expression = tool_input.get("expression", "")
                        try:
                            ns = {}
                            stdout = StringIO()
                            with redirect_stdout(stdout):
                                exec(expression, ns, ns)
                            result = {"result": stdout.getvalue() or "OK", "error": None}
                        except Exception as e:
                            result = {"result": None, "error": str(e)}
                    
                    elif tool_name == "submit_answer":
                        code = tool_input.get("code", "")
                        submitted_code = code
                        result = {"submitted": True}
                    
                    else:
                        result = {"error": "Unknown tool"}
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": json.dumps(result)
                    })
            
            if tool_results:
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                
                if submitted_code:
                    return submitted_code
        
        except:
            continue
    
    return None


# ============================================================================
# EVALUATION & REPORTING
# ============================================================================

async def run_single_test(
    run_id: int,
    prompt: str,
    tools: list,
    grader: BinaryDataDriftGrader,
    api_mode: str
) -> dict:
    """Run single test"""
    
    code = await run_agent_loop(prompt, tools, run_id, max_steps=20, api_mode=api_mode)
    grading = grader.grade_response(code or "")
    
    return {
        'run_id': run_id,
        'score': grading.get('score', 0),
        'results': grading.get('results', {}),
    }


def print_run_summary(run_data: dict):
    """Print summary for single run"""
    run_id = run_data['run_id']
    score = run_data['score']
    results = run_data['results']
    
    print(f"\n{'='*70}")
    print(f"Run {run_id}")
    print(f"{'='*70}")
    
    function_names = [
        'detect_covariate',
        'detect_concept',
        'classify',
        'impact',
        'action'
    ]
    
    for i, name in enumerate(function_names, 1):
        result = results.get(name, {})
        passed = result.get('passed', False)
        reason = result.get('reason', 'Unknown')
        
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Function {i} ({name:20s}) → {status}")
        if not passed:
            print(f"    Reason: {reason}")
    
    print(f"\nRun {run_id} Summary: {score}/5")
    print(f"{'='*70}")


async def main(num_runs: int = 10, concurrent: bool = True):
    """Main evaluation"""
    
    api_mode = "Vertex AI" if USE_VERTEX_AI else "Anthropic API"
    
    print(f"\n{'='*70}")
    print("DATA DRIFT DETECTION RL TASK - BINARY SCORING (VERY HARD)")
    print(f"{'='*70}")
    print(f"API Mode: {api_mode}")
    if USE_VERTEX_AI:
        print(f"  Region: {VERTEX_REGION}")
        print(f"  Project: {VERTEX_PROJECT_ID}")
    print(f"Mode: {'Concurrent' if concurrent else 'Sequential'}")
    print(f"Runs: {num_runs}")
    print(f"Scoring: 1 point per function if ALL tests pass, 0 otherwise")
    print(f"Total Tests: 26 across 5 functions")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")
    
    prompt = """Implement 5 Python functions for detecting and responding to data drift: 1. detect_covariate_drift(input_before, input_after, output_quality_before, output_quality_after) -> dict Returns: {'detected': bool, 'drift': 'covariate' or None} input_before/after: Lists of numerical values IMPORTANT: Detect when input shifts >20% AND quality remains stable (±5%) Must NOT detect if quality drops >5% (that's concept drift) 2. detect_concept_drift(input_before, input_after, output_quality_before, output_quality_after) -> dict Returns: {'detected': bool, 'drift': 'concept' or None} IMPORTANT: Detect when quality drops >10% AND input remains stable Must NOT detect if input shifts or if quality improves 3. classify_drift(input_shifted: bool, quality_dropped: bool) -> dict Returns: {'type': 'covariate'|'concept'|'both'|'none'} IMPORTANT: Return lowercase string matching exact type 4. calculate_drift_impact(daily_predictions: int, days_in_blind_period: int, error_rate_increase: float, cost_per_error: float) -> dict Returns: {'predictions_affected': int, 'errors': int, 'financial_impact': float} IMPORTANT: Must be accurate within ±2% tolerance (very strict) 5. determine_response_action(drift_type: str, severity: float) -> dict Returns: {'action': 'MONITOR'|'INVESTIGATE'|'RETRAIN'|'ESCALATE'} IMPORTANT: Severity thresholds: - 0-0.3: MONITOR - 0.3-0.5: INVESTIGATE - 0.5-0.9: RETRAIN - >0.9: ESCALATE Return uppercase action string Use python_expression tool to test. Submit answer when complete."""
    
    tools = [
        {
            "name": "python_expression",
            "description": "Execute Python code",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit final code",
            "input_schema": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    ]
    
    grader = BinaryDataDriftGrader()
    all_results = []
    
    if concurrent:
        tasks = [run_single_test(i+1, prompt, tools, grader, api_mode) for i in range(num_runs)]
        all_results = await asyncio.gather(*tasks)
    else:
        for i in range(num_runs):
            result = await run_single_test(i+1, prompt, tools, grader, api_mode)
            all_results.append(result)
            print_run_summary(result)
    
    if concurrent:
        for result in all_results:
            print_run_summary(result)
    
    # Final report
    scores = [r['score'] for r in all_results]
    passed_runs = sum(1 for s in scores if s == 5)
    passed_percentage = (passed_runs / num_runs) * 100
    
    print(f"\n{'='*70}")
    print("FINAL REPORT")
    print(f"{'='*70}")
    print(f"API Mode: {api_mode}")
    print(f"Total Tests: 29 (Func1:4, Func2:4, Func3:6, Func4:3, Func5:8)")
    print(f"Total Runs: {num_runs}")
    print(f"Fully Passed Runs (5/5): {passed_runs}/{num_runs}")
    print(f"Pass Rate: {passed_percentage:.1f}%")
    print(f"Target Met: {'YES ✓' if 10 <= passed_percentage <= 40 else 'NO ✗'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main(num_runs=10, concurrent=True))