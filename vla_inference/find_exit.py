import lerobot.configs.policies as p
import inspect
src = inspect.getsource(p)
for i, line in enumerate(src.split('\n')):
    if 'exit' in line.lower() or 'sys.exit' in line.lower() or 'raise' in line.lower():
        print(i, line)