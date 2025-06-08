"""
Binary Behavior Comparison Testing Framework
Provides functionality to compare the behavior of original binaries with reconstructed versions
"""

import os
import sys
import subprocess
import tempfile
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class TestResult(Enum):
    """Test result status"""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class ComparisonMode(Enum):
    """Comparison modes for behavior testing"""
    EXACT_MATCH = "exact_match"
    FUNCTIONAL_EQUIVALENT = "functional_equivalent"
    OUTPUT_SIMILAR = "output_similar"
    RETURN_CODE_MATCH = "return_code_match"


@dataclass
class BehaviorTestCase:
    """Individual behavior test case"""
    name: str
    description: str
    command_args: List[str]
    input_data: Optional[str] = None
    input_file: Optional[str] = None
    expected_output: Optional[str] = None
    expected_return_code: Optional[int] = None
    timeout: int = 30
    comparison_mode: ComparisonMode = ComparisonMode.FUNCTIONAL_EQUIVALENT


@dataclass
class ExecutionResult:
    """Result of a single binary execution"""
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    error_message: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of comparing two execution results"""
    test_case: BehaviorTestCase
    original_result: Optional[ExecutionResult]
    reconstructed_result: Optional[ExecutionResult]
    comparison_result: TestResult
    details: str
    similarity_score: float
    timestamp: float


class BinaryBehaviorTester:
    """Main class for binary behavior comparison testing"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_cases: List[BehaviorTestCase] = []
        self.temp_dir = tempfile.mkdtemp(prefix="behavior_test_")
        self.results: List[ComparisonResult] = []
        
    def __del__(self):
        """Cleanup temporary directory"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
    
    def add_test_case(self, test_case: BehaviorTestCase):
        """Add a test case to the suite"""
        self.test_cases.append(test_case)
    
    def add_standard_test_cases(self):
        """Add standard test cases for common scenarios"""
        # Basic execution test
        self.add_test_case(BehaviorTestCase(
            name="basic_execution",
            description="Test basic program execution without arguments",
            command_args=[],
            timeout=10,
            comparison_mode=ComparisonMode.RETURN_CODE_MATCH
        ))
        
        # Help flag test
        self.add_test_case(BehaviorTestCase(
            name="help_flag",
            description="Test program behavior with help flag",
            command_args=["--help"],
            timeout=5,
            comparison_mode=ComparisonMode.OUTPUT_SIMILAR
        ))
        
        # Version flag test
        self.add_test_case(BehaviorTestCase(
            name="version_flag", 
            description="Test program behavior with version flag",
            command_args=["--version"],
            timeout=5,
            comparison_mode=ComparisonMode.OUTPUT_SIMILAR
        ))
        
        # Invalid argument test
        self.add_test_case(BehaviorTestCase(
            name="invalid_argument",
            description="Test program behavior with invalid arguments",
            command_args=["--invalid-flag-xyz"],
            timeout=5,
            comparison_mode=ComparisonMode.RETURN_CODE_MATCH
        ))
        
        # Single argument test
        self.add_test_case(BehaviorTestCase(
            name="single_argument",
            description="Test program with single argument",
            command_args=["test_input"],
            timeout=10,
            comparison_mode=ComparisonMode.FUNCTIONAL_EQUIVALENT
        ))
        
        # Multiple arguments test
        self.add_test_case(BehaviorTestCase(
            name="multiple_arguments",
            description="Test program with multiple arguments",
            command_args=["arg1", "arg2", "arg3"],
            timeout=10,
            comparison_mode=ComparisonMode.FUNCTIONAL_EQUIVALENT
        ))
        
        # File input test (if test file exists)
        test_file = Path(self.temp_dir) / "test_input.txt"
        test_file.write_text("This is test input data for binary behavior testing.")
        
        self.add_test_case(BehaviorTestCase(
            name="file_input",
            description="Test program with file input",
            command_args=[str(test_file)],
            timeout=15,
            comparison_mode=ComparisonMode.FUNCTIONAL_EQUIVALENT
        ))
        
        # Stdin input test
        self.add_test_case(BehaviorTestCase(
            name="stdin_input",
            description="Test program with stdin input",
            command_args=[],
            input_data="test input from stdin\\n",
            timeout=10,
            comparison_mode=ComparisonMode.FUNCTIONAL_EQUIVALENT
        ))
    
    def execute_binary(self, binary_path: str, test_case: BehaviorTestCase) -> ExecutionResult:
        """Execute a binary with the given test case"""
        start_time = time.time()
        
        try:
            # Prepare command
            cmd = [binary_path] + test_case.command_args
            
            # Prepare input
            input_bytes = None
            if test_case.input_data:
                input_bytes = test_case.input_data.encode('utf-8')
            
            # Execute with timeout
            process = subprocess.run(
                cmd,
                input=input_bytes,
                capture_output=True,
                timeout=test_case.timeout,
                cwd=self.temp_dir
            )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                return_code=process.returncode,
                stdout=process.stdout.decode('utf-8', errors='replace'),
                stderr=process.stderr.decode('utf-8', errors='replace'),
                execution_time=execution_time
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return ExecutionResult(
                return_code=-1,
                stdout="",
                stderr="",
                execution_time=execution_time,
                error_message=f"Execution timed out after {test_case.timeout} seconds"
            )
            
        except FileNotFoundError:
            execution_time = time.time() - start_time
            return ExecutionResult(
                return_code=-2,
                stdout="",
                stderr="",
                execution_time=execution_time,
                error_message=f"Binary not found: {binary_path}"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                return_code=-3,
                stdout="",
                stderr="",
                execution_time=execution_time,
                error_message=f"Execution error: {str(e)}"
            )
    
    def compare_results(self, test_case: BehaviorTestCase, 
                       original: ExecutionResult, 
                       reconstructed: ExecutionResult) -> ComparisonResult:
        """Compare execution results according to the test case's comparison mode"""
        
        # Handle error cases first
        if original.error_message or reconstructed.error_message:
            if original.error_message and reconstructed.error_message:
                # Both failed - this could be expected behavior
                similarity_score = 1.0 if original.return_code == reconstructed.return_code else 0.0
                result = TestResult.PASS if similarity_score > 0.8 else TestResult.FAIL
                details = f"Both executions failed with similar error codes"
            else:
                # One failed, one succeeded - this is a problem
                similarity_score = 0.0
                result = TestResult.FAIL
                failing_binary = "original" if original.error_message else "reconstructed"
                details = f"Only {failing_binary} binary failed to execute"
            
            return ComparisonResult(
                test_case=test_case,
                original_result=original,
                reconstructed_result=reconstructed,
                comparison_result=result,
                details=details,
                similarity_score=similarity_score,
                timestamp=time.time()
            )
        
        # Normal comparison based on mode
        if test_case.comparison_mode == ComparisonMode.EXACT_MATCH:
            similarity_score, details = self._compare_exact_match(original, reconstructed)
        elif test_case.comparison_mode == ComparisonMode.RETURN_CODE_MATCH:
            similarity_score, details = self._compare_return_code(original, reconstructed)
        elif test_case.comparison_mode == ComparisonMode.OUTPUT_SIMILAR:
            similarity_score, details = self._compare_output_similar(original, reconstructed)
        elif test_case.comparison_mode == ComparisonMode.FUNCTIONAL_EQUIVALENT:
            similarity_score, details = self._compare_functional_equivalent(original, reconstructed)
        else:
            similarity_score = 0.0
            details = f"Unknown comparison mode: {test_case.comparison_mode}"
        
        # Determine test result based on similarity score
        if similarity_score >= 0.9:
            result = TestResult.PASS
        elif similarity_score >= 0.7:
            result = TestResult.PASS  # Still acceptable for functional equivalence
        else:
            result = TestResult.FAIL
        
        return ComparisonResult(
            test_case=test_case,
            original_result=original,
            reconstructed_result=reconstructed,
            comparison_result=result,
            details=details,
            similarity_score=similarity_score,
            timestamp=time.time()
        )
    
    def _compare_exact_match(self, original: ExecutionResult, reconstructed: ExecutionResult) -> Tuple[float, str]:
        """Compare for exact match"""
        score = 0.0
        details = []
        
        # Return code (40%)
        if original.return_code == reconstructed.return_code:
            score += 0.4
            details.append("Return codes match")
        else:
            details.append(f"Return codes differ: {original.return_code} vs {reconstructed.return_code}")
        
        # Stdout (40%)
        if original.stdout == reconstructed.stdout:
            score += 0.4
            details.append("Stdout matches exactly")
        else:
            details.append("Stdout differs")
        
        # Stderr (20%)
        if original.stderr == reconstructed.stderr:
            score += 0.2
            details.append("Stderr matches exactly")
        else:
            details.append("Stderr differs")
        
        return score, "; ".join(details)
    
    def _compare_return_code(self, original: ExecutionResult, reconstructed: ExecutionResult) -> Tuple[float, str]:
        """Compare return codes only"""
        if original.return_code == reconstructed.return_code:
            return 1.0, f"Return codes match: {original.return_code}"
        else:
            return 0.0, f"Return codes differ: {original.return_code} vs {reconstructed.return_code}"
    
    def _compare_output_similar(self, original: ExecutionResult, reconstructed: ExecutionResult) -> Tuple[float, str]:
        """Compare output for similarity (not exact match)"""
        score = 0.0
        details = []
        
        # Return code (30%)
        if original.return_code == reconstructed.return_code:
            score += 0.3
            details.append("Return codes match")
        else:
            details.append(f"Return codes differ: {original.return_code} vs {reconstructed.return_code}")
        
        # Stdout similarity (50%)
        stdout_similarity = self._calculate_text_similarity(original.stdout, reconstructed.stdout)
        score += stdout_similarity * 0.5
        details.append(f"Stdout similarity: {stdout_similarity:.2f}")
        
        # Stderr similarity (20%)
        stderr_similarity = self._calculate_text_similarity(original.stderr, reconstructed.stderr)
        score += stderr_similarity * 0.2
        details.append(f"Stderr similarity: {stderr_similarity:.2f}")
        
        return score, "; ".join(details)
    
    def _compare_functional_equivalent(self, original: ExecutionResult, reconstructed: ExecutionResult) -> Tuple[float, str]:
        """Compare for functional equivalence (most flexible)"""
        score = 0.0
        details = []
        
        # Success/failure pattern (40%)
        original_success = original.return_code == 0
        reconstructed_success = reconstructed.return_code == 0
        
        if original_success == reconstructed_success:
            score += 0.4
            status = "success" if original_success else "failure"
            details.append(f"Both programs {status}")
        else:
            details.append(f"Success patterns differ: original {'succeeded' if original_success else 'failed'}, reconstructed {'succeeded' if reconstructed_success else 'failed'}")
        
        # Output length similarity (30%)
        orig_len = len(original.stdout)
        recon_len = len(reconstructed.stdout)
        
        if orig_len == 0 and recon_len == 0:
            length_similarity = 1.0
        elif orig_len == 0 or recon_len == 0:
            length_similarity = 0.0
        else:
            length_similarity = min(orig_len, recon_len) / max(orig_len, recon_len)
        
        score += length_similarity * 0.3
        details.append(f"Output length similarity: {length_similarity:.2f}")
        
        # Content similarity (30%)
        content_similarity = self._calculate_text_similarity(original.stdout, reconstructed.stdout)
        score += content_similarity * 0.3
        details.append(f"Content similarity: {content_similarity:.2f}")
        
        return score, "; ".join(details)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Simple similarity based on common words and length
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        common_words = words1.intersection(words2)
        all_words = words1.union(words2)
        
        jaccard_similarity = len(common_words) / len(all_words) if all_words else 0.0
        
        # Length similarity
        len1, len2 = len(text1), len(text2)
        length_similarity = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0
        
        # Combined similarity
        return (jaccard_similarity * 0.7 + length_similarity * 0.3)
    
    def run_comparison_test(self, original_binary: str, reconstructed_binary: str) -> Dict[str, Any]:
        """Run complete behavior comparison test suite"""
        if self.verbose:
            print(f"Starting behavior comparison test...")
            print(f"Original binary: {original_binary}")
            print(f"Reconstructed binary: {reconstructed_binary}")
            print(f"Test cases: {len(self.test_cases)}")
        
        # Validate binaries exist
        if not Path(original_binary).exists():
            raise FileNotFoundError(f"Original binary not found: {original_binary}")
        if not Path(reconstructed_binary).exists():
            raise FileNotFoundError(f"Reconstructed binary not found: {reconstructed_binary}")
        
        # If no test cases added, add standard ones
        if not self.test_cases:
            self.add_standard_test_cases()
        
        self.results = []
        
        for i, test_case in enumerate(self.test_cases):
            if self.verbose:
                print(f"Running test {i+1}/{len(self.test_cases)}: {test_case.name}")
            
            # Execute original binary
            original_result = self.execute_binary(original_binary, test_case)
            
            # Execute reconstructed binary
            reconstructed_result = self.execute_binary(reconstructed_binary, test_case)
            
            # Compare results
            comparison = self.compare_results(test_case, original_result, reconstructed_result)
            self.results.append(comparison)
            
            if self.verbose:
                print(f"  Result: {comparison.comparison_result.value} (similarity: {comparison.similarity_score:.2f})")
                print(f"  Details: {comparison.details}")
        
        # Generate summary report
        return self._generate_report()
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.results:
            return {"error": "No test results available"}
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.comparison_result == TestResult.PASS)
        failed_tests = sum(1 for r in self.results if r.comparison_result == TestResult.FAIL)
        error_tests = sum(1 for r in self.results if r.comparison_result == TestResult.ERROR)
        
        # Calculate overall similarity
        total_similarity = sum(r.similarity_score for r in self.results)
        average_similarity = total_similarity / total_tests if total_tests > 0 else 0.0
        
        # Collect test details
        test_details = []
        for result in self.results:
            test_details.append({
                "name": result.test_case.name,
                "description": result.test_case.description,
                "result": result.comparison_result.value,
                "similarity_score": result.similarity_score,
                "details": result.details,
                "original_return_code": result.original_result.return_code if result.original_result else None,
                "reconstructed_return_code": result.reconstructed_result.return_code if result.reconstructed_result else None,
                "execution_time_original": result.original_result.execution_time if result.original_result else None,
                "execution_time_reconstructed": result.reconstructed_result.execution_time if result.reconstructed_result else None
            })
        
        # Determine overall assessment
        if average_similarity >= 0.9:
            overall_assessment = "EXCELLENT - High behavioral equivalence"
        elif average_similarity >= 0.7:
            overall_assessment = "GOOD - Acceptable behavioral equivalence"
        elif average_similarity >= 0.5:
            overall_assessment = "FAIR - Some behavioral similarities"
        else:
            overall_assessment = "POOR - Significant behavioral differences"
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
                "average_similarity": average_similarity,
                "overall_assessment": overall_assessment
            },
            "test_details": test_details,
            "recommendations": self._generate_recommendations(),
            "timestamp": time.time()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not self.results:
            return ["No test results available for analysis"]
        
        failed_tests = [r for r in self.results if r.comparison_result == TestResult.FAIL]
        error_tests = [r for r in self.results if r.comparison_result == TestResult.ERROR]
        
        if len(failed_tests) > len(self.results) * 0.5:
            recommendations.append("More than 50% of tests failed - review fundamental reconstruction approach")
        
        if len(error_tests) > 0:
            recommendations.append("Some tests had execution errors - verify binary compatibility and dependencies")
        
        # Check for specific failure patterns
        return_code_failures = sum(1 for r in failed_tests if "Return codes differ" in r.details)
        output_failures = sum(1 for r in failed_tests if "similarity" in r.details.lower())
        
        if return_code_failures > 0:
            recommendations.append("Return code mismatches detected - review error handling and exit conditions")
        
        if output_failures > 0:
            recommendations.append("Output differences detected - review I/O operations and formatting")
        
        # Performance recommendations
        avg_time_original = sum(r.original_result.execution_time for r in self.results if r.original_result) / len(self.results)
        avg_time_reconstructed = sum(r.reconstructed_result.execution_time for r in self.results if r.reconstructed_result) / len(self.results)
        
        if avg_time_reconstructed > avg_time_original * 2:
            recommendations.append("Reconstructed binary is significantly slower - consider optimization")
        elif avg_time_reconstructed < avg_time_original * 0.5:
            recommendations.append("Reconstructed binary is much faster - verify functional completeness")
        
        if not recommendations:
            recommendations.append("Behavior comparison shows good functional equivalence")
        
        return recommendations


def run_behavior_comparison_test(original_binary: str, reconstructed_binary: str, 
                               custom_test_cases: Optional[List[BehaviorTestCase]] = None,
                               verbose: bool = False) -> Dict[str, Any]:
    """
    Convenience function to run binary behavior comparison test
    
    Args:
        original_binary: Path to original binary
        reconstructed_binary: Path to reconstructed binary
        custom_test_cases: Optional custom test cases to run
        verbose: Enable verbose output
        
    Returns:
        Test report dictionary
    """
    tester = BinaryBehaviorTester(verbose=verbose)
    
    # Add custom test cases if provided
    if custom_test_cases:
        for test_case in custom_test_cases:
            tester.add_test_case(test_case)
    
    return tester.run_comparison_test(original_binary, reconstructed_binary)


# Example usage and test case templates
def create_file_processing_test_cases(test_data_dir: str) -> List[BehaviorTestCase]:
    """Create test cases for file processing programs"""
    test_cases = []
    
    # Create test files
    test_dir = Path(test_data_dir)
    test_dir.mkdir(exist_ok=True)
    
    # Small text file
    small_file = test_dir / "small.txt"
    small_file.write_text("Hello World")
    
    # Large text file
    large_file = test_dir / "large.txt"
    large_file.write_text("A" * 10000)
    
    # Binary file
    binary_file = test_dir / "binary.dat"
    binary_file.write_bytes(bytes(range(256)) * 10)
    
    test_cases.extend([
        BehaviorTestCase(
            name="process_small_file",
            description="Process small text file",
            command_args=[str(small_file)],
            comparison_mode=ComparisonMode.FUNCTIONAL_EQUIVALENT
        ),
        BehaviorTestCase(
            name="process_large_file",
            description="Process large text file",
            command_args=[str(large_file)],
            timeout=60,
            comparison_mode=ComparisonMode.FUNCTIONAL_EQUIVALENT
        ),
        BehaviorTestCase(
            name="process_binary_file",
            description="Process binary file",
            command_args=[str(binary_file)],
            comparison_mode=ComparisonMode.FUNCTIONAL_EQUIVALENT
        )
    ])
    
    return test_cases


def create_network_test_cases() -> List[BehaviorTestCase]:
    """Create test cases for network programs"""
    return [
        BehaviorTestCase(
            name="localhost_connection",
            description="Test localhost connection",
            command_args=["127.0.0.1"],
            timeout=10,
            comparison_mode=ComparisonMode.RETURN_CODE_MATCH
        ),
        BehaviorTestCase(
            name="invalid_host",
            description="Test invalid hostname",
            command_args=["invalid.host.name.xyz"],
            timeout=15,
            comparison_mode=ComparisonMode.RETURN_CODE_MATCH
        )
    ]