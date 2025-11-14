"""
Validation script - runs all tests in sequence
"""
import subprocess
import sys


def run_test(script_name, description):
    """Run a test script and report results"""
    print("\n" + "="*60)
    print(f"Running: {description}")
    print("="*60)

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            check=True,
            timeout=30
        )
        print(f"[PASS] {description}")
        return True
    except subprocess.CalledProcessError:
        print(f"[FAIL] {description}")
        return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description}")
        return False
    except Exception as e:
        print(f"[ERROR] {description}: {e}")
        return False


def main():
    """Run all validation tests"""
    print("\n" + "="*60)
    print("CLABSIGuard Validation Suite")
    print("="*60)

    tests = [
        ("test_camera.py", "Camera & CUDA Test"),
        ("clabsi_guard.py", "Model Architecture Test"),
        ("monitor.py", "Compliance Monitor Test"),
        ("benchmark.py", "Performance Benchmark"),
    ]

    results = []

    for script, description in tests:
        passed = run_test(script, description)
        results.append((description, passed))

    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    for description, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "[+]" if passed else "[-]"
        print(f"{symbol} {description:40s} {status}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    print("="*60)
    print(f"Results: {passed_count}/{total} tests passed")
    print("="*60)

    if passed_count == total:
        print("\n[SUCCESS] All validation tests PASSED!")
        print("System ready for webcam demo.")
        print("\nRun: python webcam_demo.py")
    else:
        print("\n[FAILURE] Some tests FAILED. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
