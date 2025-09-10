import gradio as gr
import requests
import base64
import uuid
import os
from PIL import Image
from io import BytesIO
import json

# ---------- Helper functions ----------

def image_to_base64(img_file):
    """Convert uploaded file to base64 (JPEG only)."""
    image = Image.open(img_file).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def save_base64_to_jpeg(b64_data):
    """Decode base64 image and save as unique JPEG file."""
    img_bytes = base64.b64decode(b64_data)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    filename = f"generated_{uuid.uuid4().hex}.jpeg"
    img.save(filename, format="JPEG")
    return filename

# ---------- Main API call ----------

def gemini_generate(api_key, prompt, influencer_img, product_img):
    if not api_key:
        return "❌ Error: API key required", None, "", "{}"

    try:
        # Convert both images to base64
        influencer_b64 = image_to_base64(influencer_img)
        product_b64 = image_to_base64(product_img)

        # Build correct payload
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": influencer_b64
                            }
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": product_b64
                            }
                        }
                    ]
                }
            ]
        }

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        # Send request
        resp = requests.post(url, headers=headers, json=payload)
        print("HTTP status:", resp.status_code)
        print("Response snippet:", resp.text[:1000]) # debug

        if resp.status_code != 200:
            return f"❌ API Error {resp.status_code}", None, "", json.dumps(resp.json(), indent=2)

        data = resp.json()

        # Extract image result
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        img_b64 = None
        source_used = None
        mime_type = None

        for part in parts:
            if "inlineData" in part:
                img_b64 = part["inlineData"].get("data")
                mime_type = part["inlineData"].get("mimeType")
                source_used = "inlineData"
                break
            elif "fileData" in part:
                file_uri = part["fileData"]["fileUri"]
                mime_type = part["fileData"].get("mimeType", "image/jpeg")
                r = requests.get(file_uri)
                img_b64 = base64.b64encode(r.content).decode("utf-8")
                source_used = "fileData"
                break

        if not img_b64:
            return "⚠️ No image found in response", None, "", json.dumps(data, indent=2)

        # Save as JPEG
        filename = save_base64_to_jpeg(img_b64)
        with open(filename, "rb") as f:
            final_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Public URL (works in Colab/Gradio)
        final_url = f"/file={filename}"

        # Metadata
        metadata = {
            "modelVersion": data.get("modelVersion", "unknown"),
            "responseId": data.get("responseId", "unknown"),
            "mimeType": mime_type,
            "source": source_used,
            "finalImage": final_url
        }

        return "✅ Success", filename, final_b64, json.dumps(metadata, indent=2)

    except Exception as e:
        return f"❌ Exception: {str(e)}", None, "", "{}"

# ---------- Gradio UI ----------

with gr.Blocks(css=".gradio-container {max-width: 900px; margin: auto;}") as demo:
    gr.Markdown("## Gemini Image Preview Demo (Influencer + Product)")

    with gr.Row():
        api_key = gr.Textbox(label="Google API Key", type="text", placeholder="Enter your x-goog-api-key")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Describe the product placement idea...")
    with gr.Row():
        influencer = gr.Image(type="filepath", label="Influencer Image", height=260, width=260)
        product = gr.Image(type="filepath", label="Product Image", height=260, width=260)

    generate_btn = gr.Button("🚀 Generate")

    with gr.Row():
        status = gr.Textbox(label="Status")
    with gr.Row():
        output_img = gr.Image(label="Preview (JPEG)")
    with gr.Row():
        output_b64 = gr.Textbox(label="Base64 (JPEG)", lines=6)
    with gr.Row():
        metadata = gr.Textbox(label="Metadata", lines=8)

    generate_btn.click(
        gemini_generate,
        inputs=[api_key, prompt, influencer, product],
        outputs=[status, output_img, output_b64, metadata]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
