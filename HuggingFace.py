import requests

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
headers = {"Authorization": f"Bearer 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"}


def query(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()


def hugging_face(image_path):
    output = query(image_path)

    return output[0]['generated_text']


print(hugging_face(image_path))