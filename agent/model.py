from langchain_ollama import OllamaLLM

def get_model():
    return OllamaLLM(model="qwen3:0.6b")

# All the code below is for running on GPU
# However, Qwen3-0.6B is a small enough model to run on CPU with Ollama

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from langchain.schema import HumanMessage, AIMessage
#
# class Qwen3LLM:
#     """Simple GPU-backed Qwen3-0.6B LLM, LangChain-style interface."""
#
#     def __init__(self, model_name="Qwen/Qwen3-0.6B", device=None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             dtype=torch.float16
#         ).to(self.device)
#
#     def invoke(self, messages, max_tokens=4096, temperature=0.7):
#         """
#         messages: list of HumanMessage / AIMessage
#         Returns: tuple (thinking_content, content)
#         """
#         # Convert messages to dict format for chat template
#         chat_messages = [{"role": "user", "content": m.content} for m in messages]
#
#         # Apply chat template with thinking block
#         text = self.tokenizer.apply_chat_template(
#             chat_messages,
#             tokenize=False,
#             add_generation_prompt=True,
#             enable_thinking=True  # adds <think> at the beginning
#         )
#
#         # Tokenize and send to device
#         model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
#
#         # Generate output
#         generated_ids = self.model.generate(
#             **model_inputs,
#             max_new_tokens=max_tokens,
#             do_sample=False,
#             temperature=temperature
#         )
#
#         # Remove input ids from output to get generated portion
#         output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
#
#         # Split thinking content from normal content
#         try:
#             # 151668 = </think> token id
#             index = len(output_ids) - output_ids[::-1].index(151668)
#         except ValueError:
#             index = 0
#
#         thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
#         content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
#
#         return thinking_content + content
#
#     def __call__(self, prompt: str, max_tokens=4096, temperature=0.7):
#         """Optional: call the model directly with a string."""
#         thinking, content = self.invoke([HumanMessage(content=prompt)], max_tokens=max_tokens, temperature=temperature)
#         return thinking + content
