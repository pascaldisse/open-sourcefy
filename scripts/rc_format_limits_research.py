#!/usr/bin/env python3
"""
RC Format Limitations Research Script

This script tests Windows RC compiler with progressively larger string tables
to identify practical limits for STRINGTABLE entries and document memory/performance impacts.

Research findings will help determine viable approaches for embedding 22,317 strings.
"""

import os
import sys
import time
import subprocess
import tempfile
import json
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class RCLimitsResearcher:
    """Research RC format limitations with systematic testing."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_dir = self.project_root / "temp" / "rc_limits_research"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # RC compiler path (Windows SDK)
        self.rc_exe = Path("/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe")
        
        # Test configuration
        self.test_sizes = [100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 22317]
        self.results = []
        
    def generate_test_rc_file(self, string_count: int, test_id: str) -> Path:
        """Generate RC file with specified number of strings."""
        rc_path = self.test_dir / f"test_{test_id}_{string_count}.rc"
        
        with open(rc_path, 'w', encoding='utf-8') as f:
            f.write(f"// RC Limits Test - {string_count} strings\n")
            f.write("#include \"resource.h\"\n\n")
            f.write("STRINGTABLE\nBEGIN\n")
            
            # Generate test strings with varying content
            for i in range(string_count):
                string_id = 1000 + i
                # Create diverse string content to test different scenarios
                if i % 4 == 0:
                    content = f"TestString_{i:06d}_Short"
                elif i % 4 == 1:
                    content = f"TestString_{i:06d}_MediumLength_WithSpecialChars_@#$%^&*()"
                elif i % 4 == 2:
                    content = f"TestString_{i:06d}_VeryLongString_" + "A" * 100
                else:
                    content = f"Unicode_String_{i:06d}_測試字符串_тестовая строка"
                
                # Escape quotes and handle special characters
                escaped_content = content.replace('"', '\\"').replace('\\', '\\\\')
                f.write(f'    {string_id}, "{escaped_content}"\n')
            
            f.write("END\n")
            
        return rc_path
    
    def generate_resource_header(self, string_count: int) -> Path:
        """Generate resource.h header file."""
        header_path = self.test_dir / "resource.h"
        
        with open(header_path, 'w') as f:
            f.write("// Generated resource header\n")
            f.write("#ifndef RESOURCE_H\n")
            f.write("#define RESOURCE_H\n\n")
            
            # Define string IDs
            for i in range(string_count):
                string_id = 1000 + i
                f.write(f"#define IDS_STRING_{i:06d} {string_id}\n")
            
            f.write("\n#endif // RESOURCE_H\n")
        
        return header_path
    
    def compile_rc_file(self, rc_path: Path, string_count: int) -> Dict:
        """Compile RC file and measure performance metrics."""
        print(f"Testing RC compilation with {string_count} strings...")
        
        # Generate output paths
        res_path = rc_path.with_suffix('.res')
        
        # Convert Linux paths to Windows paths for RC compiler
        rc_win_path = str(rc_path).replace('/mnt/c/', 'C:\\').replace('/', '\\')
        res_win_path = str(res_path).replace('/mnt/c/', 'C:\\').replace('/', '\\')
        
        # Start timing and memory monitoring
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        try:
            # Run RC compiler with Windows paths
            cmd = [
                str(self.rc_exe),
                "/fo", res_win_path,
                rc_win_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(self.test_dir)
            )
            
            end_time = time.time()
            compilation_time = end_time - start_time
            final_memory = process.memory_info().rss
            memory_delta = final_memory - initial_memory
            
            # Check if compilation succeeded
            success = result.returncode == 0 and res_path.exists()
            
            # Get file sizes
            rc_size = rc_path.stat().st_size if rc_path.exists() else 0
            res_size = res_path.stat().st_size if res_path.exists() else 0
            
            test_result = {
                "string_count": string_count,
                "success": success,
                "compilation_time": compilation_time,
                "memory_delta_mb": memory_delta / (1024 * 1024),
                "rc_file_size_mb": rc_size / (1024 * 1024),
                "res_file_size_mb": res_size / (1024 * 1024),
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
            
            if success:
                print(f"✅ SUCCESS: {string_count} strings compiled in {compilation_time:.2f}s")
                print(f"   RC size: {rc_size/1024:.1f}KB, RES size: {res_size/1024:.1f}KB")
            else:
                print(f"❌ FAILED: {string_count} strings failed compilation")
                print(f"   Error: {result.stderr[:200]}...")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: {string_count} strings compilation timed out")
            return {
                "string_count": string_count,
                "success": False,
                "compilation_time": 300,
                "memory_delta_mb": 0,
                "rc_file_size_mb": 0,
                "res_file_size_mb": 0,
                "return_code": -1,
                "stdout": "",
                "stderr": "Compilation timed out after 300 seconds",
                "command": " ".join(cmd)
            }
        
        except Exception as e:
            print(f"💥 ERROR: {string_count} strings compilation failed with exception: {e}")
            return {
                "string_count": string_count,
                "success": False,
                "compilation_time": 0,
                "memory_delta_mb": 0,
                "rc_file_size_mb": 0,
                "res_file_size_mb": 0,
                "return_code": -2,
                "stdout": "",
                "stderr": str(e),
                "command": " ".join(cmd)
            }
    
    def run_comprehensive_tests(self) -> Dict:
        """Run comprehensive RC limits testing."""
        print("🔬 Starting RC Format Limitations Research")
        print("=" * 60)
        
        # Check if RC compiler exists
        if not self.rc_exe.exists():
            print(f"❌ RC compiler not found at: {self.rc_exe}")
            return {"error": "RC compiler not available"}
        
        print(f"✅ RC compiler found: {self.rc_exe}")
        print(f"📁 Test directory: {self.test_dir}")
        print()
        
        # Test each size incrementally
        for string_count in self.test_sizes:
            print(f"📊 Testing {string_count:,} strings...")
            
            # Generate test files
            test_id = f"limit_test_{string_count}"
            self.generate_resource_header(string_count)
            rc_path = self.generate_test_rc_file(string_count, test_id)
            
            # Compile and measure
            test_result = self.compile_rc_file(rc_path, string_count)
            self.results.append(test_result)
            
            # If compilation failed, analyze why
            if not test_result["success"]:
                print(f"🚨 LIMIT REACHED: Failed at {string_count:,} strings")
                self.analyze_failure(test_result)
                
                # Try to find the exact limit between last success and this failure
                if self.results and len(self.results) > 1:
                    last_success = None
                    for prev_result in reversed(self.results[:-1]):
                        if prev_result["success"]:
                            last_success = prev_result["string_count"]
                            break
                    
                    if last_success:
                        print(f"🔍 Searching for exact limit between {last_success:,} and {string_count:,}")
                        self.binary_search_limit(last_success, string_count)
                
                break  # Stop at first failure
            
            print()
        
        # Generate comprehensive report
        return self.generate_research_report()
    
    def binary_search_limit(self, low: int, high: int) -> Optional[int]:
        """Binary search to find exact limit."""
        print(f"🔍 Binary searching limit between {low:,} and {high:,} strings...")
        
        while high - low > 1:
            mid = (low + high) // 2
            print(f"   Testing {mid:,} strings...")
            
            # Generate and test
            test_id = f"binary_search_{mid}"
            self.generate_resource_header(mid)
            rc_path = self.generate_test_rc_file(mid, test_id)
            result = self.compile_rc_file(rc_path, mid)
            
            if result["success"]:
                low = mid
                print(f"   ✅ {mid:,} succeeded")
            else:
                high = mid
                print(f"   ❌ {mid:,} failed")
        
        print(f"🎯 Exact limit found: {low:,} strings maximum")
        return low
    
    def analyze_failure(self, test_result: Dict) -> None:
        """Analyze why compilation failed."""
        print("🔍 Failure Analysis:")
        print(f"   Return code: {test_result['return_code']}")
        print(f"   Error output: {test_result['stderr'][:500]}")
        
        # Check for specific error patterns
        stderr = test_result['stderr'].lower()
        if 'memory' in stderr or 'out of memory' in stderr:
            print("   💭 Failure appears to be memory-related")
        elif 'too many' in stderr or 'limit' in stderr:
            print("   📏 Failure appears to be due to limits")
        elif 'timeout' in stderr:
            print("   ⏰ Failure due to timeout")
        else:
            print("   ❓ Unknown failure cause")
    
    def generate_research_report(self) -> Dict:
        """Generate comprehensive research report."""
        report = {
            "research_metadata": {
                "timestamp": time.time(),
                "test_count": len(self.results),
                "rc_compiler": str(self.rc_exe),
                "test_directory": str(self.test_dir)
            },
            "test_results": self.results,
            "analysis": self.analyze_results(),
            "recommendations": self.generate_recommendations()
        }
        
        # Save report to file
        report_path = self.test_dir / "rc_limits_research_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Research report saved: {report_path}")
        return report
    
    def analyze_results(self) -> Dict:
        """Analyze test results to identify patterns and limits."""
        if not self.results:
            return {"error": "No test results available"}
        
        successful_tests = [r for r in self.results if r["success"]]
        failed_tests = [r for r in self.results if not r["success"]]
        
        analysis = {
            "total_tests": len(self.results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
        }
        
        if successful_tests:
            max_successful = max(r["string_count"] for r in successful_tests)
            analysis["max_successful_strings"] = max_successful
            
            # Performance metrics for successful tests
            times = [r["compilation_time"] for r in successful_tests]
            memory_usage = [r["memory_delta_mb"] for r in successful_tests]
            
            analysis["performance"] = {
                "min_compile_time": min(times),
                "max_compile_time": max(times),
                "avg_compile_time": sum(times) / len(times),
                "max_memory_delta_mb": max(memory_usage),
                "avg_memory_delta_mb": sum(memory_usage) / len(memory_usage)
            }
        
        if failed_tests:
            min_failed = min(r["string_count"] for r in failed_tests)
            analysis["min_failed_strings"] = min_failed
            
            # Analyze failure patterns
            failure_reasons = {}
            for test in failed_tests:
                stderr = test["stderr"].lower()
                if 'memory' in stderr:
                    failure_reasons["memory"] = failure_reasons.get("memory", 0) + 1
                elif 'timeout' in stderr:
                    failure_reasons["timeout"] = failure_reasons.get("timeout", 0) + 1
                elif 'limit' in stderr:
                    failure_reasons["limit"] = failure_reasons.get("limit", 0) + 1
                else:
                    failure_reasons["unknown"] = failure_reasons.get("unknown", 0) + 1
            
            analysis["failure_patterns"] = failure_reasons
        
        return analysis
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not self.results:
            return ["No test results available for recommendations"]
        
        successful_tests = [r for r in self.results if r["success"]]
        
        if successful_tests:
            max_strings = max(r["string_count"] for r in successful_tests)
            
            if max_strings >= 22317:
                recommendations.append("✅ RC format can handle 22,317 strings - proceed with single RC file approach")
            elif max_strings >= 10000:
                recommendations.append(f"🔶 RC format can handle {max_strings:,} strings - use segmented approach with ~{22317//max_strings + 1} RC files")
            elif max_strings >= 5000:
                recommendations.append(f"🟡 RC format limited to {max_strings:,} strings - use segmented approach with ~{22317//max_strings + 1} RC files")
            else:
                recommendations.append(f"🔴 RC format severely limited to {max_strings:,} strings - consider binary injection approach")
            
            # Performance recommendations
            max_time = max(r["compilation_time"] for r in successful_tests)
            if max_time > 60:
                recommendations.append("⏰ Long compilation times detected - implement parallel RC compilation")
            
            max_memory = max(r["memory_delta_mb"] for r in successful_tests)
            if max_memory > 500:
                recommendations.append("💭 High memory usage detected - implement memory-efficient processing")
        
        # Always recommend researching binary injection as alternative
        recommendations.append("🔧 Research PE manipulation libraries (LIEF, pefile) for direct resource injection as alternative")
        recommendations.append("🔀 Consider hybrid approach: RC for smaller string sets + binary injection for large sets")
        
        return recommendations

def main():
    """Main research execution."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "/mnt/c/Users/pascaldisse/Downloads/open-sourcefy"
    
    researcher = RCLimitsResearcher(project_root)
    
    try:
        report = researcher.run_comprehensive_tests()
        
        if "error" not in report:
            print("\n" + "=" * 60)
            print("🎯 RESEARCH SUMMARY")
            print("=" * 60)
            
            analysis = report["analysis"]
            if "max_successful_strings" in analysis:
                print(f"📊 Maximum successful strings: {analysis['max_successful_strings']:,}")
            
            if "performance" in analysis:
                perf = analysis["performance"]
                print(f"⏱️  Compilation time range: {perf['min_compile_time']:.2f}s - {perf['max_compile_time']:.2f}s")
                print(f"💭 Memory usage range: {perf['avg_memory_delta_mb']:.1f}MB avg, {perf['max_memory_delta_mb']:.1f}MB max")
            
            print("\n🎯 RECOMMENDATIONS:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"{i}. {rec}")
        
    except KeyboardInterrupt:
        print("\n🛑 Research interrupted by user")
    except Exception as e:
        print(f"💥 Research failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()