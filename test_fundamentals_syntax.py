"""
Syntax check for fundamentals module (doesn't require yfinance installed).
"""

import sys
import ast

files_to_check = [
    'src/fundamentals/__init__.py',
    'src/fundamentals/metrics.py',
    'src/fundamentals/earnings.py',
    'src/fundamentals/health_score.py',
    'src/fundamentals/analyzer.py',
    'app/pages/8_Fundamentals.py',
]

print("=" * 80)
print("SYNTAX CHECK FOR FUNDAMENTALS MODULE")
print("=" * 80)

all_passed = True

for file_path in files_to_check:
    print(f"\nChecking: {file_path}")
    try:
        with open(file_path, 'r') as f:
            code = f.read()

        # Parse the code to check for syntax errors
        ast.parse(code)
        print(f"  ✓ Syntax valid")

    except SyntaxError as e:
        print(f"  ✗ SYNTAX ERROR: {e}")
        all_passed = False
    except FileNotFoundError:
        print(f"  ✗ FILE NOT FOUND")
        all_passed = False
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("✓ ALL SYNTAX CHECKS PASSED!")
    print("\nThe fundamentals module is syntactically correct.")
    print("Runtime testing requires yfinance to be installed:")
    print("  pip install yfinance")
else:
    print("✗ SOME CHECKS FAILED")
    sys.exit(1)

print("=" * 80)
