import psutil
ram = psutil.virtual_memory()
print(f"Total RAM: {ram.total / 1e9:.1f} GB")
print(f"Available RAM: {ram.available / 1e9:.1f} GB")
print(f"Used RAM: {ram.used / 1e9:.1f} GB")
print(f"Percent used: {ram.percent}%")