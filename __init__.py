from .azure_flux_node import (
    FluxKontextImageToImage,
    FluxKontextTextToImage,
    Flux11ProTextToImage,
)

NODE_CLASS_MAPPINGS = {
    "FluxKontextImageToImage": FluxKontextImageToImage,
    "FluxKontextTextToImage": FluxKontextTextToImage,
    "Flux11ProTextToImage": Flux11ProTextToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKontextImageToImage": "FLUX.1-Kontext-ImageToImage",
    "FluxKontextTextToImage": "FLUX.1-Kontext-TextToImage",
    "Flux11ProTextToImage": "FLUX-1.1-pro-TextToImage",
}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'FluxKontextImageToImage',
    'FluxKontextTextToImage',
    'Flux11ProTextToImage',
]
