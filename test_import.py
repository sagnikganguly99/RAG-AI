import sys
print("Python version:", sys.version)

try:
    import langchain
    print("✓ langchain imported successfully")
except Exception as e:
    print(f"✗ langchain error: {e}")

try:
    from openai import OpenAI
    print("✓ OpenAI imported successfully")
except Exception as e:
    print(f"✗ OpenAI error: {e}")

print("Ready to run chunking.py")
