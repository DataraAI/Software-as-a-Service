from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(tags='lerobot', limit=20)
for m in models:
    print(m.id)
