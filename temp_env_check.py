import sys
import os
import importlib.util

print('exe', sys.executable)
print('cwd', os.getcwd())
print('path0', sys.path[0])
print('sys.path')
for p in sys.path:
    print('  ', p)
print('PYTHONPATH', os.environ.get('PYTHONPATH'))
print('pandas spec', importlib.util.find_spec('pandas'))
try:
    import pandas
    print('pandas file', pandas.__file__)
except Exception as e:
    print('import error', type(e).__name__, e)
