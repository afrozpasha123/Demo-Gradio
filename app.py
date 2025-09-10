import gradio as gr
import requests
import base64
import uuid
import os
import json
from PIL import Image
from io import BytesIO

# ---------- Helper functions ----------

def image_to_base64(img_file):
    """Convert uploaded file to base64 (JPEG only)."""
    image = Image.open(img_file).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ---------- Main API call ----------

def gemini_generate_base64(api_key, prompt, influencer_img, product_img):
    if not api_key:
        return "❌ Error: API key required", ""

    try:
        influencer_b64 = image_to_base64(influencer_img)
        product_b64 = image_to_base64(product_img)

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

        resp = requests.post(url, headers=headers, json=payload)
        print("HTTP status:", resp.status_code)
        print("Response snippet:", resp.text[:1000])  # Debug log

        if resp.status_code != 200:
            return f"❌ API Error {resp.status_code}", json.dumps(resp.json(), indent=2)

        data = resp.json()

        # Extract base64 directly
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        img_b64 = None
        for part in parts:
            if "inlineData" in part:
                img_b64 = part["inlineData"].get("data")
                break

        if not img_b64:
            return "⚠️ No base64 image found in response", json.dumps(data, indent=2)

        # Print first 200 chars in logs
        print("Base64 output (first 200 chars):", img_b64[:200])

        return "✅ Success", img_b64

    except Exception as e:
        return f"❌ Exception: {str(e)}", ""

# ---------- Gradio UI ----------

with gr.Blocks(css=".gradio-container {max-width: 900px; margin: auto;}") as demo:
    gr.Markdown("## Gemini Base64 Output Demo (Colab)")

    with gr.Row():
        api_key = gr.Textbox(label="Google API Key", type="text", placeholder="Enter your x-goog-api-key")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Describe the product placement idea...")
    with gr.Row():
        influencer = gr.Image(type="filepath", label="Influencer Image")
        product = gr.Image(type="filepath", label="Product Image")

    generate_btn = gr.Button("🚀 Generate")

    with gr.Row():
        status = gr.Textbox(label="Status")
    with gr.Row():
        output_b64 = gr.Textbox(label="Base64 Output", lines=12)

    generate_btn.click(
        gemini_generate_base64,
        inputs=[api_key, prompt, influencer, product],
        outputs=[status, output_b64]
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)
