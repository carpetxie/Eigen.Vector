import torch
from PIL import Image
from transformers import AutoModelForCausalLM

# Specify the local path to the model directory
local_model_path = r"C:\Users\prapa\Documents\GitHub\Eigen.Vector\Ovis2-2B"

# Load the model
model = AutoModelForCausalLM.from_pretrained(local_model_path,
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# single-image input
# image_path = r"C:\Users\prapa\Documents\GitHub\Eigen.Vector\ocr\example.png"
# images = [Image.open(image_path)]
# max_partition = 9
# text = 'Describe the image.'
# query = f'<image>\n{text}'

math_background = "AI/ML Researcher"
category_query = 1 
category_text = "Does this have anything to do with PDEs?" #custom question
if category_query == 1:
    category_query = "You are an expert teacher with the purpose of clearly, concisely explaining though concepts thoroughly and step-by-step. Explain the concepts and ideas described in the attached image from the basics of my educational background, building up to the concepts employed, and a discussion of what is going on in the image itself. "
elif category_query == 2: 
    category_query = "Write a python script to visaulize what is described in the attached image."
elif category_query == 3:
    category_query = category_text

query = "Write down the text in the attached image. " + str(category_query)
cot_instructions = "Provide a step-by-step solution to the problem, and conclude with 'Now, my thinking is over. I'll tell my student the following:' followed by the final solution. My educational/skill level is " + str(math_background) + "; adjust your responses to frame from what I know to what is in the image."

## cot-style input
cot_suffix = cot_instructions
image_path = r"C:\Users\prapa\Documents\GitHub\Eigen.Vector\ocr\example.png"
images = [Image.open(image_path)]
max_partition = 9
text = query
query = f'<image>\n{text}\n{cot_suffix}'

# format conversation
prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
input_ids = input_ids.unsqueeze(0).to(device=model.device)
attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
if pixel_values is not None:
    pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
pixel_values = [pixel_values]

# generate output
with torch.inference_mode():
    gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=True,
        top_p=50,
        top_k=0.9,
        temperature=0.7,
        repetition_penalty=None,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=text_tokenizer.pad_token_id,
        use_cache=True
    )
    output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
    output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f'Output:\n{output}')