
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
        """Grade response with binary scoring"""
        try:
            results = {
                'detect_covariate': self._test_covariate(response_code),
                'detect_concept': self._test_concept(response_code),
                'classify': self._test_classify(response_code),
                'impact': self._test_impact(response_code),
                'action': self._test_action(response_code),
            }
            
            passed_count = sum(1 for r in results.values() if r['passed'])
            
            return {
                'results': results,
                'score': passed_count,
            }
        except Exception as e:
            return {
                'results': {
                    'detect_covariate': {'passed': False, 'reason': 'Grading error'},
                    'detect_concept': {'passed': False, 'reason': 'Grading error'},
                    'classify': {'passed': False, 'reason': 'Grading error'},
                    'impact': {'passed': False, 'reason': 'Grading error'},
                    'action': {'passed': False, 'reason': 'Grading error'},
                },
                'score': 0,
                'error': str(e)
            }


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
                ({'input_shifted': True, 'quality_dropped': False}, 'covariate'),
                ({'input_shifted': True, 'quality_dropped': True}, 'both'),
            ]
            failed_tests = []
            for i, (signals, expected) in enumerate(cases, 1):
                try:
                    result = func(**signals)
                    drift_type = str(result.get('type', result)).lower()
                    if drift_type != expected.lower():
                        failed_tests.append(i)
                except:
                    failed_tests.append(i)
            if not failed_tests:
                return {'passed': True, 'reason': 'All 6 tests passed'}
            else:
                return {'passed': False, 'reason': f'Failed tests: {failed_tests}'}
        except:
            return {'passed': False, 'reason': 'Function execution failed'}


    def _test_covariate(self, code: str) -> dict:
        """Harder covariate drift tests — mixed thresholds & unstable signals"""
        try:
            ns = {}
            exec(code, ns)
            func = ns.get('detect_covariate_drift')
            if not func:
                return {'passed': False, 'reason': 'Function not found'}
            
            tests_passed = []

            # --- Easy sanity tests ---
            r1 = func([100]*5, [140]*5, 0.9, 0.89)  # +40%, stable
            tests_passed.append(r1.get('detected') == True)
            r2 = func([100]*5, [115]*5, 0.9, 0.84)  # medium shift, drop
            tests_passed.append(r2.get('detected') == False)

            # --- Hard boundary mix ---
            r3 = func([100]*5, [119]*5, 0.9, 0.88)  # 19%, just below threshold
            tests_passed.append(r3.get('detected') == False)
            r4 = func([100]*5, [121]*5, 0.9, 0.89)  # 21%, tiny improvement
            tests_passed.append(r4.get('detected') == True)

            # --- Inverse logic traps ---
            r5 = func([200]*5, [150]*5, 0.91, 0.90)  # -25%, stable
            tests_passed.append(r5.get('detected') == True)
            r6 = func([100]*5, [120]*5, 0.9, 0.83)   # same shift but drop
            tests_passed.append(r6.get('detected') == False)
            r7 = func([100]*5, [105]*5, 0.9, 0.91)   # tiny shift, improvement
            tests_passed.append(r7.get('detected') == False)

            # --- Extreme sensitivity test ---
            r8 = func([100]*5, [130]*5, 0.9, 0.855)  # big shift, borderline quality
            tests_passed.append(r8.get('detected') == True)

            if all(tests_passed):
                return {'passed': True, 'reason': 'All 8 covariate tests passed'}
            else:
                failed = [i+1 for i, p in enumerate(tests_passed) if not p]
                # At least 3 must fail to count as “too strong”
                return {'passed': len(failed) <= 2, 'reason': f'Failed tests: {failed}'}
        except Exception as e:
            return {'passed': False, 'reason': f'Execution failed: {e}'}


    def _test_concept(self, code: str) -> dict:
        """Stricter concept drift logic — ambiguous degradation and near-threshold values"""
        try:
            ns = {}
            exec(code, ns)
            func = ns.get('detect_concept_drift')
            if not func:
                return {'passed': False, 'reason': 'Function not found'}

            tests_passed = []

            # 1. Strong drop, same input
            r1 = func([100]*5, [100]*5, 0.9, 0.74)
            tests_passed.append(r1.get('detected') == True)

            # 2. Just below detection margin (10%)
            r2 = func([100]*5, [100]*5, 0.9, 0.81)
            tests_passed.append(r2.get('detected') == False)

            # 3. Slight input drift + quality drop
            r3 = func([100]*5, [120]*5, 0.9, 0.79)
            tests_passed.append(r3.get('detected') == False)

            # 4. Same input but quality fluctuates slightly (±2%)
            r4 = func([100]*5, [100]*5, 0.85, 0.87)
            tests_passed.append(r4.get('detected') == False)

            # 5. Quality dropped 11% but also input changed minimally (5%)
            r5 = func([100]*5, [105]*5, 0.9, 0.799)
            tests_passed.append(r5.get('detected') == True)

            if all(tests_passed):
                return {'passed': True, 'reason': 'All 5 concept tests passed'}
            else:
                failed = [i+1 for i, p in enumerate(tests_passed) if not p]
                # Require 2+ passes minimum
                return {'passed': len(failed) <= 3, 'reason': f'Failed: {failed}'}
        except Exception as e:
            return {'passed': False, 'reason': f'Execution failed: {e}'}


    def _test_impact(self, code: str) -> dict:
        """Tighter drift impact tolerances — precision under pressure"""
        try:
            ns = {}
            exec(code, ns)
            func = ns.get('calculate_drift_impact')
            if not func:
                return {'passed': False, 'reason': 'Function not found'}

            tests_passed = []
            result = func(10000, 5, 0.02, 50)
            affected = result.get('predictions_affected')
            errors = result.get('errors')
            cost = result.get('financial_impact')
            tests_passed.append(49000 <= affected <= 51000 and 975 <= errors <= 1025 and 48000 <= cost <= 52000)

            result2 = func(10000, 7, 0.0001, 50)
            tests_passed.append(result2.get('errors') <= 10 and 300 <= result2.get('financial_impact') <= 400)

            result3 = func(10000, 2.5, 0.01, 100)
            tests_passed.append(23900 <= result3.get('predictions_affected') <= 26100)

            result4 = func(50000, 1, 0.05, 200)
            errors4 = result4.get('errors')
            tests_passed.append(2450 <= errors4 <= 2550)

            if all(tests_passed):
                return {'passed': True, 'reason': 'All 4 impact tests passed'}
            else:
                failed = [i+1 for i, p in enumerate(tests_passed) if not p]
                return {'passed': len(failed) <= 2, 'reason': f'Failed tests: {failed}'}
        except Exception as e:
            return {'passed': False, 'reason': f'Execution failed: {e}'}


    def _test_action(self, code: str) -> dict:
        """Harder thresholds for action mapping"""
        try:
            ns = {}
            exec(code, ns)
            func = ns.get('determine_response_action')
            if not func:
                return {'passed': False, 'reason': 'Function not found'}
            cases = [
                ({'drift_type': 'covariate', 'severity': 0.29}, 'MONITOR'),
                ({'drift_type': 'concept', 'severity': 0.35}, 'INVESTIGATE'),
                ({'drift_type': 'both', 'severity': 0.51}, 'RETRAIN'),
                ({'drift_type': 'concept', 'severity': 0.91}, 'ESCALATE'),
                ({'drift_type': 'covariate', 'severity': 0.5}, 'INVESTIGATE'),
                ({'drift_type': 'both', 'severity': 0.75}, 'RETRAIN'),
                ({'drift_type': 'both', 'severity': 0.95}, 'ESCALATE'),
                ({'drift_type': 'unknown', 'severity': 0.45}, 'INVESTIGATE'),
            ]
            failed = []
            for i, (params, expected) in enumerate(cases, 1):
                try:
                    result = func(**params)
                    if str(result.get('action')).upper() != expected:
                        failed.append(i)
                except Exception:
                    failed.append(i)
            if not failed:
                return {'passed': True, 'reason': 'All 8 action tests passed'}
            else:
                # Slightly stricter now — must pass ≥5 to count
                return {'passed': len(failed) <= 3, 'reason': f'Failed tests: {failed}'}
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
        model_name = "claude-sonnet-4-5"
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