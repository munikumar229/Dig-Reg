#!/usr/bin/env python3
"""
Simple test runner script for the Dig-Reg project.
Runs all available tests and reports results.
"""

import os
import sys
import subprocess
from pathlib import Path

def get_python_executable():
    """Get the correct Python executable path."""
    # First try the current Python executable (works in venv)
    return sys.executable

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n🧪 {description}")
    print("=" * 60)
    
    # Replace 'python' with the actual Python executable
    python_exe = get_python_executable()
    command = command.replace('python', python_exe, 1)
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=Path(__file__).parent.parent
        )
        print(result.stdout)
        if result.stderr:
            print("⚠️ Warnings:", result.stderr)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print("Error output:", e.stderr)
        if e.stdout:
            print("Standard output:", e.stdout)
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Main test runner function."""
    print("🚀 Dig-Reg Test Suite")
    print("=" * 60)
    
    # Check if we're in the project root
    project_root = Path(__file__).parent.parent
    if not (project_root / "requirements.txt").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # List of tests to run
    tests = [
        ("python tests/test_models.py", "Model Functionality Tests"),
        ("python tests/test_backend_api.py", "Backend API Tests"),
        ("python scripts/process_data.py", "Data Processing Test"),
    ]
    
    # Optional tests (run if files exist)
    optional_tests = [
        ("python tests/test_parameters.py", "Parameter Tests"),
    ]
    
    results = []
    
    # Run mandatory tests
    for command, description in tests:
        if "test_backend_api.py" in command:
            # Check if backend API test exists
            if not (project_root / "tests" / "test_backend_api.py").exists():
                print(f"\n⏭️ Skipping {description} (file not found)")
                continue
        
        success = run_command(command, description)
        results.append((description, success))
    
    # Run optional tests
    for command, description in optional_tests:
        test_file = project_root / command.split()[-1]
        if test_file.exists():
            success = run_command(command, description)
            results.append((description, success))
        else:
            print(f"\n⏭️ Skipping {description} (optional, file not found)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description}: {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The project is ready for deployment.")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review and fix.")
        sys.exit(1)

if __name__ == "__main__":
    main()