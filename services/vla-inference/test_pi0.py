import traceback
try:
    from lerobot.policies.pi0 import PI0Policy
    print("Importing PI0Policy... OK")
    print("Calling from_pretrained...")
    policy = PI0Policy.from_pretrained("lerobot/pi0_base")
    print("SUCCESS:", policy)
except SystemExit as e:
    print(f"SystemExit caught: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Exception caught: {e}")
    traceback.print_exc()