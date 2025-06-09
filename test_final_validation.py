#!/usr/bin/env python3
"""
Test script for Final Validation Orchestrator
Tests the automated final validation system with original and recompiled binaries.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.final_validation_orchestrator import FinalValidationOrchestrator


async def main():
    """Main test function for final validation system"""
    
    print("🏆 Testing Final Validation Orchestrator System")
    print("=" * 60)
    
    # Define paths
    original_binary = Path("/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/input/launcher.exe")
    recompiled_binary = Path("/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/compilation/launcher_rebuilt.exe")
    output_dir = Path("/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/reports")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if binaries exist
    print(f"📁 Checking binary files...")
    print(f"   Original binary: {original_binary}")
    print(f"   Exists: {'✅ YES' if original_binary.exists() else '❌ NO'}")
    if original_binary.exists():
        print(f"   Size: {original_binary.stat().st_size:,} bytes")
    
    print(f"   Recompiled binary: {recompiled_binary}")
    print(f"   Exists: {'✅ YES' if recompiled_binary.exists() else '❌ NO'}")
    if recompiled_binary.exists():
        print(f"   Size: {recompiled_binary.stat().st_size:,} bytes")
    
    print(f"   Output directory: {output_dir}")
    print()
    
    if not original_binary.exists():
        print("❌ ERROR: Original binary not found!")
        return
    
    if not recompiled_binary.exists():
        print("❌ ERROR: Recompiled binary not found!")
        print("   Please run the Matrix pipeline first to generate the recompiled binary.")
        return
    
    # Initialize the validation orchestrator
    print("🔧 Initializing Final Validation Orchestrator...")
    orchestrator = FinalValidationOrchestrator()
    print("✅ Orchestrator initialized")
    print()
    
    # Execute final validation
    print("🚀 Starting Final Validation Process...")
    print("   This will execute 10 validation tasks to measure recompilation accuracy")
    print()
    
    try:
        # Run the final validation
        validation_report = await orchestrator.execute_final_validation(
            original_binary=original_binary,
            recompiled_binary=recompiled_binary,
            output_dir=output_dir
        )
        
        # Display results
        print("📊 FINAL VALIDATION RESULTS")
        print("=" * 60)
        
        final_validation = validation_report["final_validation"]
        total_match = final_validation["total_match_percentage"]
        success = final_validation["success"]
        execution_time = final_validation["execution_time"]
        
        print(f"🎯 Overall Match Percentage: {total_match:.2f}%")
        print(f"🏁 Final Status: {'✅ SUCCESS' if success else '⚠️  NEEDS IMPROVEMENT'}")
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print()
        
        # Display individual task results
        print("📋 INDIVIDUAL TASK RESULTS")
        print("-" * 60)
        
        for i, task in enumerate(validation_report["task_results"], 1):
            status_icon = "✅" if task["success"] else "❌"
            print(f"{i:2d}. {task['task']}")
            print(f"    Status: {status_icon} {'PASS' if task['success'] else 'FAIL'}")
            print(f"    Match:  {task['match_percentage']:6.1f}%")
            print(f"    Details: {task['details']}")
            print()
        
        # Display binary comparison summary
        print("🔍 BINARY COMPARISON SUMMARY")
        print("-" * 60)
        
        binary_comp = validation_report["binary_comparison"]
        print(f"Original Size:    {binary_comp['original_size']:,} bytes")
        print(f"Recompiled Size:  {binary_comp['recompiled_size']:,} bytes")
        print(f"Size Match:       {binary_comp['size_match']:.1f}%")
        print(f"Hash Match:       {binary_comp['hash_match']:.1f}%")
        print()
        
        # Performance assessment
        print("🏆 PERFECT BINARY RECOMPILATION ASSESSMENT")
        print("-" * 60)
        
        if total_match >= 99.9:
            print("🥇 EXTRAORDINARY! Near-perfect binary recompilation achieved!")
            print("   Your recompiled binary is virtually identical to the original.")
        elif total_match >= 95.0:
            print("🥈 EXCELLENT! Very high quality binary recompilation!")
            print("   Minor differences exist but overall recompilation is highly successful.")
        elif total_match >= 85.0:
            print("🥉 GOOD! Solid binary recompilation with room for improvement.")
            print("   The recompiled binary captures most of the original structure.")
        elif total_match >= 70.0:
            print("⚠️  MODERATE: Basic recompilation achieved but significant gaps remain.")
            print("   Further optimization of the Matrix pipeline is recommended.")
        else:
            print("🔧 DEVELOPING: Early stage recompilation with substantial work needed.")
            print("   Continue refining the decompilation and reconstruction process.")
        
        print()
        print("📄 Detailed reports saved to:")
        print(f"   JSON: {output_dir}/final_validation_report.json")
        print(f"   HTML: {output_dir}/final_validation_report.html")
        print()
        
        # Next steps recommendation
        print("🔮 NEXT STEPS RECOMMENDATIONS")
        print("-" * 60)
        
        failing_tasks = [task for task in validation_report["task_results"] if not task["success"]]
        
        if not failing_tasks:
            print("🎉 All validation tasks passed! Consider:")
            print("   • Running additional test cases with different binaries")
            print("   • Implementing byte-perfect timestamp preservation")
            print("   • Adding automated checksum correction")
        else:
            print("🎯 Priority improvements needed:")
            for task in failing_tasks[:3]:  # Show top 3 failing tasks
                print(f"   • {task['task']} ({task['match_percentage']:.1f}% match)")
            
            if len(failing_tasks) > 3:
                print(f"   • ... and {len(failing_tasks) - 3} other tasks")
        
        print()
        print("✅ Final Validation Test Complete!")
        
    except Exception as e:
        print(f"❌ ERROR during validation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())