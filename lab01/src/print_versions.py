import sys 
import platform 
from importlib.metadata import distributions

print("Python version:", sys.version)
print("Platform:", platform.platform())
print("\nInstalled packages:")
for dist in sorted(distributions(), key=lambda d: d.metadata['Name'].lower()):
    print(f"{dist.metadata['Name']} == {dist.version}")