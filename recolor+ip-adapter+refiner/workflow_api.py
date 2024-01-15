import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np
from PIL import Image
from pathlib import Path

ROOT = Path("/mnt/d/sairoid_skin_color_dataset")

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    CLIPTextEncode,
    ControlNetLoader,
    CheckpointLoaderSimple,
    EmptyLatentImage,
    ControlNetApplyAdvanced,
    SaveImage,
    NODE_CLASS_MAPPINGS,
    ImageScale,
    LoadImage,
    KSamplerAdvanced,
    VAEDecode,
    CLIPVisionLoader,
    VAEEncode,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sdxl/animeIllustDiffusion_v06.safetensors"
        )

        emptylatentimage = EmptyLatentImage()
        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text="anime, 1girl, white background, close up face, looking at viewer",
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="(embedding:negativeXL_D:1.0), (embedding:unaestheticXL_Sky3.1:1.0), text, signature",
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        checkpointloadersimple_12 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_refiner_1.0.safetensors"
        )

        cliptextencode_15 = cliptextencode.encode(
            text="anime, 1girl, white background, close up face, looking at viewer",
            clip=get_value_at_index(checkpointloadersimple_12, 1),
        )

        cliptextencode_16 = cliptextencode.encode(
            text="(embedding:negativeXL_D:1.0), (embedding:unaestheticXL_Sky3.1:1.0), text, signature",
            clip=get_value_at_index(checkpointloadersimple_12, 1),
        )

        controlnetloader = ControlNetLoader()
        controlnetloader_49 = controlnetloader.load_controlnet(
            control_net_name="sd_control_collection/sai_xl_recolor_256lora.safetensors"
        )

        loadimage = LoadImage()

        ipadaptermodelloader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
        ipadaptermodelloader_58 = ipadaptermodelloader.load_ipadapter_model(
            ipadapter_file="ip-adapter_sdxl_vit-h.bin"
        )
        clipvisionloader = CLIPVisionLoader()
        clipvisionloader_60 = clipvisionloader.load_clip(
            clip_name="sd15_clip.safetensors"
        )

        imagescale = ImageScale()
        vaeencode = VAEEncode()

        imagescaletototalpixels = NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
        ipadapterapply = NODE_CLASS_MAPPINGS["IPAdapterApply"]()
        controlnetapplyadvanced = ControlNetApplyAdvanced()
        ksampleradvanced = KSamplerAdvanced()
        vaedecode = VAEDecode()

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(width=1024, height=1024, batch_size=1)
        for avator_idx in range(3):
            for skincolor_idx in range(5):
        #        motif_img = torch.from_numpy(motif_arr[None, :, :, :].astype(np.float32) / 255.0)
        #        guide_img = torch.from_numpy(guide_arr[None, :, :, :].astype(np.float32) / 255.0)

                avator_loadimage = (torch.from_numpy(np.array(Image.open(ROOT/"avator"/f"{avator_idx}.png").convert("RGB"))[None, :, :, :].astype(np.float32) / 255.0), None)
                skincolor_loadimage = (torch.from_numpy(np.array(Image.open(ROOT/"skin_color_specification"/f"{skincolor_idx}.png").convert("RGB"))[None, :, :, :].astype(np.float32) / 255.0), None)

#                avator_loadimage = loadimage.load_image(image="input_0.png") # avator
#                skincolor_loadimage = loadimage.load_image(image="skin32.png") # skincolor
                #torch.Size([1, 1000, 1013, 3]) -> [0]
        #        print(f"\x1b[31m{avator_loadimage[2].shape}\x1b[0m")

#                import IPython; IPython.embed()
#                raise Exception()

## vae {
#                skincolor_imagescale_62 = imagescale.upscale(
#                    upscale_method="nearest-exact",
#                    width=1024,
#                    height=1024,
#                    crop="disabled",
#                    image=get_value_at_index(skincolor_loadimage, 0),
#                )

#                skincolor_vaeencode_61 = vaeencode.encode(
#                    pixels=get_value_at_index(skincolor_imagescale_62, 0),
#                    vae=get_value_at_index(checkpointloadersimple_4, 2),
#                )
## vae }


                for q in range(5):
                    imagescaletototalpixels_57 = imagescaletototalpixels.upscale(
                        upscale_method="nearest-exact",
                        megapixels=1,
                        image=get_value_at_index(skincolor_loadimage, 0),
                    )

                    ipadapterapply_59 = ipadapterapply.apply_ipadapter(
                        weight=1,
                        noise=0,
                        ipadapter=get_value_at_index(ipadaptermodelloader_58, 0),
                        clip_vision=get_value_at_index(clipvisionloader_60, 0),
                        image=get_value_at_index(imagescaletototalpixels_57, 0),
                        model=get_value_at_index(checkpointloadersimple_4, 0),
                    )

                    imagescaletototalpixels_51 = imagescaletototalpixels.upscale(
                        upscale_method="nearest-exact",
                        megapixels=1,
                        image=get_value_at_index(avator_loadimage, 0),
                    )

                    controlnetapplyadvanced_50 = controlnetapplyadvanced.apply_controlnet(
                        strength=1,
                        start_percent=0,
                        end_percent=1,
                        positive=get_value_at_index(cliptextencode_6, 0),
                        negative=get_value_at_index(cliptextencode_7, 0),
                        control_net=get_value_at_index(controlnetloader_49, 0),
                        image=get_value_at_index(imagescaletototalpixels_51, 0),
                    )

                    ksampleradvanced_10 = ksampleradvanced.sample(
                        add_noise="enable",
                        noise_seed=random.randint(1, 2**64),
                        steps=25,
                        cfg=8,
                        sampler_name="euler",
                        scheduler="normal",
                        start_at_step=0,
                        end_at_step=20,
                        return_with_leftover_noise="enable",
                        model=get_value_at_index(ipadapterapply_59, 0),
                        positive=get_value_at_index(controlnetapplyadvanced_50, 0),
                        negative=get_value_at_index(controlnetapplyadvanced_50, 1),
                        latent_image=get_value_at_index(emptylatentimage_5, 0),
                    )

                    ksampleradvanced_11 = ksampleradvanced.sample(
                        add_noise="disable",
                        noise_seed=random.randint(1, 2**64),
                        steps=25,
                        cfg=8,
                        sampler_name="euler",
                        scheduler="normal",
                        start_at_step=20,
                        end_at_step=10000,
                        return_with_leftover_noise="disable",
                        model=get_value_at_index(checkpointloadersimple_12, 0),
                        positive=get_value_at_index(cliptextencode_15, 0),
                        negative=get_value_at_index(cliptextencode_16, 0),
                        latent_image=get_value_at_index(ksampleradvanced_10, 0),
                    )

                    decoded_img = vaedecode.decode(
                        samples=get_value_at_index(ksampleradvanced_11, 0),
                        vae=get_value_at_index(checkpointloadersimple_12, 2),
                    )
                    output  = (255*decoded_img[0][0]).to(torch.uint8).numpy()
                    Image.fromarray(output).save(ROOT/"output"/f"avator{avator_idx}_skincolor{skincolor_idx}_variation{q}.png")


if __name__ == "__main__":
    main()
