# Python Environment Diagnostic Script
# This will help identify why mlflow import is failing

import sys
import os
import subprocess

print("=" * 60)
print("PYTHON ENVIRONMENT DIAGNOSTIC")
print("=" * 60)

# 1. Check Python executable being used
print(f"1. Python executable: {sys.executable}")
print(f"2. Python version: {sys.version}")
print(f"3. Python path: {sys.path}")

print("\n" + "-" * 40)
print("INSTALLED PACKAGES CHECK")
print("-" * 40)

# 4. Check installed packages
try:
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                          capture_output=True, text=True)
    print("4. Installed packages:")
    for line in result.stdout.split('\n'):
        if 'mlflow' in line.lower() or 'scikit' in line.lower() or 'pandas' in line.lower() or 'numpy' in line.lower():
            print(f"   {line}")
except Exception as e:
    print(f"4. Error checking packages: {e}")

print("\n" + "-" * 40)
print("IMPORT TESTS")
print("-" * 40)

# 5. Test imports one by one
packages_to_test = ['mlflow', 'sklearn', 'pandas', 'numpy']

for package in packages_to_test:
    try:
        __import__(package)
        print(f"5. ✅ {package} imports successfully")
    except ImportError as e:
        print(f"5. ❌ {package} import failed: {e}")
    except Exception as e:
        print(f"5. ⚠️  {package} unexpected error: {e}")

print("\n" + "-" * 40)
print("ENVIRONMENT VARIABLES")
print("-" * 40)

# 6. Check environment variables
print(f"6. PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"7. PATH: {os.environ.get('PATH', 'Not set')[:200]}...")
print(f"8. Current working directory: {os.getcwd()}")

print("\n" + "-" * 40)
print("VIRTUAL ENVIRONMENT CHECK")
print("-" * 40)

# 9. Check if in virtual environment
print(f"9. Virtual environment: {os.environ.get('VIRTUAL_ENV', 'Not detected')}")
print(f"10. Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'Not detected')}")

print("\n" + "-" * 40)
print("PIP LOCATION CHECK")
print("-" * 40)

# 11. Check pip location
try:
    pip_result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                              capture_output=True, text=True)
    print(f"11. Pip version and location: {pip_result.stdout.strip()}")
except Exception as e:
    print(f"11. Error checking pip: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
print("\nNext steps based on results:")
print("1. If mlflow shows as installed but import fails:")
print("   - There's likely a Python environment mismatch")
print("2. If different Python paths are shown:")
print("   - You're running script with different Python than where mlflow is installed")
print("3. If virtual environment is not detected but you think you're in one:")
print("   - Virtual environment might not be properly activated")