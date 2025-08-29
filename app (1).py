import gradio as gr
import numpy as np
import random
import torch
import spaces
import os
import json

from PIL import Image
from diffusers import QwenImageEditPipeline, FlowMatchEulerDiscreteScheduler

from huggingface_hub import InferenceClient
import math

from optimization import optimize_pipeline_
from qwenimage.pipeline_qwen_image_edit import QwenImageEditPipeline as QwenImageEditPipelineCustom
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

# --- Prompt Enhancement using Hugging Face InferenceClient ---
def polish_prompt_hf(original_prompt, system_prompt):
    """
    Rewrites the prompt using a Hugging Face InferenceClient.
    """
    # Ensure HF_TOKEN is set
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("Warning: HF_TOKEN not set. Falling back to original prompt.")
        return original_prompt

    try:
        # Initialize the client
        client = InferenceClient(
            provider="cerebras",
            api_key=api_key,
        )

        # Format the messages for the chat completions API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original_prompt}
        ]

        # Call the API
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507",
            messages=messages,
        )
        
        # Parse the response
        result = completion.choices[0].message.content
        
        # Try to extract JSON if present
        if '{"Rewritten"' in result:
            try:
                # Clean up the response
                result = result.replace('```json', '').replace('```', '')
                result_json = json.loads(result)
                polished_prompt = result_json.get('Rewritten', result)
            except:
                polished_prompt = result
        else:
            polished_prompt = result
            
        polished_prompt = polished_prompt.strip().replace("\n", " ")
        return polished_prompt
        
    except Exception as e:
        print(f"Error during API call to Hugging Face: {e}")
        # Fallback to original prompt if enhancement fails
        return original_prompt


def polish_prompt(prompt, img):
    """
    Main function to polish prompts for image editing using HF inference.
    """
    SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  

Please strictly follow the rewriting rules below:

## 1. General Principles
- Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the edited input image's overall scene.  

## 2. Task Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: "Add an animal"  
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  

### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes " ". Do not translate or alter the original language of the text, and do not change the capitalization.  
- **For text replacement tasks, always use the fixed template:**
    - Replace "xx" to "yy".  
    - Replace the xx bounding box to "yy".  
- If the user does not specify text content, infer and add concise text based on the instruction and the input image's context. For example:  
    > Original: "Add a line of text" (poster)  
    > Rewritten: "Add text "LIMITED EDITION" at the top center with slight shadow"  
- Specify text position, color, and layout in a concise way.  

### 3. Human Editing Tasks
- Maintain the person's core visual consistency (ethnicity, gender, age, hairstyle, expression, outfit, etc.).  
- If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.  
- **For expression changes, they must be natural and subtle, never exaggerated.**  
- If deletion is not specifically emphasized, the most important subject in the original image (e.g., a person, an animal) should be preserved.
    - For background change tasks, emphasize maintaining subject consistency at first.  
- Example:  
    > Original: "Change the person's hat"  
    > Rewritten: "Replace the man's hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged"  

### 4. Style Transformation or Enhancement Tasks
- If a style is specified, describe it concisely with key visual traits. For example:  
    > Original: "Disco style"  
    > Rewritten: "1970s disco: flashing lights, disco ball, mirrored walls, colorful tones"  
- If the instruction says "use reference style" or "keep current style," analyze the input image, extract main features (color, composition, texture, lighting, art style), and integrate them concisely.  
- **For coloring tasks, including restoring old photos, always use the fixed template:** "Restore old photograph, remove scratches, reduce noise, enhance details, high resolution, realistic, natural skin tones, clear facial features, no distortion, vintage photo restoration"  
- If there are other changes, place the style description at the end.

## 3. Rationality and Logic Checks
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" should be logically corrected.  
- Add missing key information: if position is unspecified, choose a reasonable area based on composition (near subject, empty space, center/edges).  

# Output Format
Return only the rewritten instruction text directly, without JSON formatting or any other wrapper.
'''
    
    # Note: We're not actually using the image in the HF version, 
    # but keeping the interface consistent
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:"
    
    return polish_prompt_hf(full_prompt, SYSTEM_PROMPT)


# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Scheduler configuration for Lightning
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# Initialize scheduler with Lightning config
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

# Load the edit pipeline with Lightning scheduler
pipe = QwenImageEditPipelineCustom.from_pretrained(
    "Qwen/Qwen-Image-Edit", 
    scheduler=scheduler,
    torch_dtype=dtype
).to(device)

# Load Lightning LoRA weights for acceleration
try:
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning", 
        weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors"
    )
    pipe.fuse_lora()
    print("Successfully loaded Lightning LoRA weights")
except Exception as e:
    print(f"Warning: Could not load Lightning LoRA weights: {e}")
    print("Continuing with base model...")

# Apply the same optimizations from the first version
pipe.transformer.__class__ = QwenImageTransformer2DModel
pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

# --- Ahead-of-time compilation ---
optimize_pipeline_(pipe, image=Image.new("RGB", (1024, 1024)), prompt="prompt")

# --- UI Constants and Helpers ---
MAX_SEED = np.iinfo(np.int32).max

# --- Main Inference Function ---
@spaces.GPU(duration=60)
def infer(
    image,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=8,  # Default to 8 steps for fast inference
    rewrite_prompt=True,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generates an edited image using the Qwen-Image-Edit pipeline with Lightning acceleration.
    """
    # Hardcode the negative prompt as in the original
    negative_prompt = " "
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"Original prompt: '{prompt}'")
    print(f"Negative Prompt: '{negative_prompt}'")
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}")
    
    if rewrite_prompt:
        prompt = polish_prompt(prompt, image)
        print(f"Rewritten Prompt: {prompt}")

    # Generate the edited image - always generate just 1 image
    try:
        images = pipe(
            image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=1  # Always generate only 1 image
        ).images
        
        # Return the first (and only) image
        return images[0], seed
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise e

# --- Examples and UI Layout ---
examples = [
    # You can add example pairs of [image_path, prompt] here
    # ["path/to/image1.jpg", "Replace the background with a beach scene"],
    # ["path/to/image2.jpg", "Add a red hat to the person"],
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#logo-title {
    text-align: center;
}
#logo-title img {
    width: 400px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
        <div id="logo-title">
            <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image Edit Logo" width="400" style="display: block; margin: 0 auto;">
            <h2 style="font-style: italic;color: #5b47d1;margin-top: -27px !important;margin-left: 96px">Fast, 8-steps with Lightning LoRA</h2>
        </div>
        """)
        gr.Markdown("""
        [Learn more](https://github.com/QwenLM/Qwen-Image) about the Qwen-Image series. 
        This demo uses the [Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) LoRA for accelerated inference.
        Try on [Qwen Chat](https://chat.qwen.ai/), or [download model](https://huggingface.co/Qwen/Qwen-Image-Edit) to run locally with ComfyUI or diffusers.
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image", 
                    show_label=True, 
                    type="pil"
                )
            # Changed from Gallery to Image
            result = gr.Image(
                label="Result", 
                show_label=True, 
                type="pil"
            )
            
        with gr.Row():
            prompt = gr.Text(
                label="Edit Instruction",
                show_label=False,
                placeholder="Describe the edit instruction (e.g., 'Replace the background with a sunset', 'Add a red hat', 'Remove the person')",
                container=False,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                true_guidance_scale = gr.Slider(
                    label="True guidance scale",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=4,
                    maximum=28,
                    step=1,
                    value=8
                )
                
            # Removed num_images_per_prompt slider entirely
            rewrite_prompt = gr.Checkbox(
                label="Enhance prompt (using HF Inference)", 
                value=True
            )

        # gr.Examples(examples=examples, inputs=[input_image, prompt], outputs=[result, seed], fn=infer, cache_examples=False)

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            input_image,
            prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
            rewrite_prompt,
            # Removed num_images_per_prompt from inputs
        ],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch()