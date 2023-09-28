import sys
from huggingface_hub import InferenceClient

if 1:
    client = InferenceClient()
    image = client.text_to_image(" ".join(sys.argv[1:]))
    image.save("img/bstronaut.png")
if 0:
    # Initialize client for a specific model
    client = InferenceClient(model="prompthero/openjourney-v4")
    client.text_to_image(" ".join(sys.argv[1:]), output_path="img/a.png")
    # Or use a generic client but pass your model as an argument
    #client = InferenceClient()
    #client.text_to_image(" ".join(*sys.argv[1:]), model="prompthero/openjourney-v4",
    #                     output_path="img/b.png")
    pass