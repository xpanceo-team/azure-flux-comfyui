import requests
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np
import json
import re


class _FluxImagesAPI:
    """
    Shared Azure OpenAI Images (Flux) helpers used by specific nodes.
    Endpoints:
      {endpoint}/openai/deployments/{deployment}/images/edits?api-version={api_version}
      {endpoint}/openai/deployments/{deployment}/images/generations?api-version={api_version}

    Returns: IMAGE tensor [N, H, W, C], float32 [0..1]
    """

    def _normalize_endpoint(self, endpoint: str) -> str:
        return endpoint if endpoint.endswith("/") else (endpoint + "/")

    def _build_edits_url(self, endpoint: str, deployment: str, api_version: str) -> str:
        endpoint = self._normalize_endpoint(endpoint)
        return f"{endpoint}openai/deployments/{deployment}/images/edits?api-version={api_version}"

    def _build_generations_url(self, endpoint: str, deployment: str, api_version: str) -> str:
        endpoint = self._normalize_endpoint(endpoint)
        return f"{endpoint}openai/deployments/{deployment}/images/generations?api-version={api_version}"

    def _tensor_to_pil(self, tensor):
        if tensor is None:
            return None
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        arr = tensor.detach().cpu().numpy()
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        img = Image.fromarray(arr)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _pil_to_tensor(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_array)[None, ...]

    def _pil_list_to_batched_tensor(self, images):
        tensors = [self._pil_to_tensor(img) for img in images]
        return torch.cat(tensors, dim=0)

    def _run(self,
             mode: str,
             azure_endpoint: str,
             deployment: str,
             api_version: str,
             api_key: str,
             editing_prompt: str,
             image_1=None,
             size: str = "1024x1024",
             n: int = 1,
             output_format: str = "png"):

        use_endpoint = azure_endpoint
        use_deployment = deployment

        if not api_key or not api_key.strip():
            raise ValueError("Please enter your Azure OpenAI API key.")
        if not editing_prompt or not editing_prompt.strip():
            raise ValueError("Please provide a non-empty prompt.")
        if not isinstance(n, int) or n < 1:
            n = 1

        size_str = (size or "").strip().lower()
        if not re.match(r"^\d{2,5}x\d{2,5}$", size_str):
            raise ValueError("Parameter 'size' must be 'WIDTHxHEIGHT', e.g. '1024x1024'.")

        headers_common = {
            "Api-Key": api_key.strip(),
        }

        if mode == "edit":
            if image_1 is None:
                raise ValueError("In 'edit' mode, 'image_1' must be provided.")
            edits_url = self._build_edits_url(use_endpoint, use_deployment, api_version)
            print(f"🚀 [Azure Images: EDIT] {edits_url}")
            print(f"📝 Prompt: {editing_prompt}")
            print(f"🧰 n={n}, size={size_str}, output_format={output_format}, model={use_deployment}")

            pil_img = self._tensor_to_pil(image_1)
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)

            files = {
                "image": ("input.png", buf, "image/png"),
            }
            data = {
                "model": use_deployment,
                "prompt": editing_prompt,
                "n": str(int(n)),
                "size": size_str,
                "output_format": output_format,
            }

            try:
                resp = requests.post(
                    edits_url,
                    headers=headers_common,
                    data=data,
                    files=files,
                    timeout=180,
                )
            except requests.RequestException as e:
                raise ValueError(f"Network error calling Azure Images Edits: {e}")
        else:
            generations_url = self._build_generations_url(use_endpoint, use_deployment, api_version)
            print(f"🚀 [Azure Images: GENERATE] {generations_url}")
            print(f"📝 Prompt: {editing_prompt}")
            print(f"🧰 n={n}, size={size_str}, output_format={output_format}, model={use_deployment}")

            payload = {
                "model": use_deployment,
                "prompt": editing_prompt,
                "n": n,
                "size": size_str,
                "output_format": output_format,
            }

            headers = {**headers_common, "Content-Type": "application/json"}

            try:
                resp = requests.post(
                    generations_url,
                    headers=headers,
                    json=payload,
                    timeout=180,
                )
            except requests.RequestException as e:
                raise ValueError(f"Network error calling Azure Images Generations: {e}")

        if resp.status_code != 200:
            txt = resp.text
            try:
                txt = json.dumps(resp.json(), ensure_ascii=False)
            except Exception:
                pass
            if resp.status_code in (400, 404) and mode == "edit" and "1.1" in use_deployment:
                txt += " — Note: This model/deployment may not support edits. Try mode='generate'."
            raise ValueError(f"Azure Images API failed ({resp.status_code}): {txt}")

        try:
            payload = resp.json()
        except Exception as e:
            raise ValueError(f"Failed to parse Azure response JSON: {e}")

        data_list = payload.get("data")
        if not data_list or not isinstance(data_list, list):
            raise ValueError(f"Unexpected Azure response structure: {json.dumps(payload)[:500]}")

        out_images = []
        for i, item in enumerate(data_list):
            b64_img = item.get("b64_json")
            if not b64_img:
                raise ValueError(f"Missing 'b64_json' at index {i}.")
            try:
                img_bytes = base64.b64decode(b64_img)
                img = Image.open(BytesIO(img_bytes))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                out_images.append(img)
            except Exception as e:
                raise ValueError(f"Failed to decode image {i}: {e}")

        batched = self._pil_list_to_batched_tensor(out_images)
        print(f"🎉 Success! Received {batched.shape[0]} image(s).")
        return (batched,)


class FluxKontextImageToImage(_FluxImagesAPI):
    """FLUX.1-Kontext — Image-to-Image (edits)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "azure_endpoint": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "https://<your-resource>.services.ai.azure.com/"
                }),
                "api_version": ("STRING", {
                    "multiline": False,
                    "default": "2025-04-01-preview",
                    "placeholder": "Azure OpenAI API version"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Enter your Azure OpenAI API key"
                }),
                "editing_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Put the person into a white t-shirt",
                    "placeholder": "Describe the edit you want"
                }),
                "image_1": ("IMAGE",),
            },
            "optional": {
                "size": ("STRING", {
                    "multiline": False,
                    "default": "1024x1024",
                    "placeholder": "WIDTHxHEIGHT, e.g. 1024x1024"
                }),
                "n": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "image/azure"
    DESCRIPTION = "Azure OpenAI Images (Flux.1-Kontext) — image-to-image (edits)"

    def run(self, azure_endpoint, api_version, api_key, editing_prompt, image_1,
            size="1024x1024", n=1, output_format="png"):
        return self._run(
            mode="edit",
            azure_endpoint=azure_endpoint,
            deployment="FLUX.1-Kontext-pro",
            api_version=api_version,
            api_key=api_key,
            editing_prompt=editing_prompt,
            image_1=image_1,
            size=size,
            n=n,
            output_format=output_format,
        )


class FluxKontextTextToImage(_FluxImagesAPI):
    """FLUX.1-Kontext — Text-to-Image (generations)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "azure_endpoint": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "https://<your-resource>.services.ai.azure.com/"
                }),
                "api_version": ("STRING", {
                    "multiline": False,
                    "default": "2025-04-01-preview",
                    "placeholder": "Azure OpenAI API version"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Enter your Azure OpenAI API key"
                }),
                "editing_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A photorealistic portrait of a person in a white t-shirt",
                    "placeholder": "Describe the image you want"
                }),
            },
            "optional": {
                "size": ("STRING", {
                    "multiline": False,
                    "default": "1024x1024",
                    "placeholder": "WIDTHxHEIGHT, e.g. 1024x1024"
                }),
                "n": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "image/azure"
    DESCRIPTION = "Azure OpenAI Images (Flux.1-Kontext) — text-to-image (generations)"

    def run(self, azure_endpoint, api_version, api_key, editing_prompt,
            size="1024x1024", n=1, output_format="png"):
        return self._run(
            mode="generate",
            azure_endpoint=azure_endpoint,
            deployment="FLUX.1-Kontext-pro",
            api_version=api_version,
            api_key=api_key,
            editing_prompt=editing_prompt,
            image_1=None,
            size=size,
            n=n,
            output_format=output_format,
        )


class Flux11ProTextToImage(_FluxImagesAPI):
    """FLUX-1.1-pro — Text-to-Image (generations)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "azure_endpoint": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "https://<your-resource>.services.ai.azure.com/"
                }),
                "api_version": ("STRING", {
                    "multiline": False,
                    "default": "2025-04-01-preview",
                    "placeholder": "Azure OpenAI API version"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Enter your Azure OpenAI API key"
                }),
                "editing_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A photorealistic portrait of a person in a white t-shirt",
                    "placeholder": "Describe the image you want"
                }),
            },
            "optional": {
                "size": ("STRING", {
                    "multiline": False,
                    "default": "1024x1024",
                    "placeholder": "WIDTHxHEIGHT, e.g. 1024x1024"
                }),
                "n": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "image/azure"
    DESCRIPTION = "Azure OpenAI Images (FLUX-1.1-pro) — text-to-image (generations)"

    def run(self, azure_endpoint, api_version, api_key, editing_prompt,
            size="1024x1024", n=1, output_format="png"):
        return self._run(
            mode="generate",
            azure_endpoint=azure_endpoint,
            deployment="FLUX-1.1-pro",
            api_version=api_version,
            api_key=api_key,
            editing_prompt=editing_prompt,
            image_1=None,
            size=size,
            n=n,
            output_format=output_format,
        )


# For repos that import node classes directly from this module
__all__ = [
    "FluxKontextImageToImage",
    "FluxKontextTextToImage",
    "Flux11ProTextToImage",
]
