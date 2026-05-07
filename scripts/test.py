from vllm import LLM

# For generative models (task=generate) only
llm = LLM(model= "mistralai/Mistral-7B-Instruct-v0.3", task="generate")  # Name or path of your model
output = llm.generate("Hello, my name is")
print(output)



# use uv
#uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl --torch-backend cpu

# use pip
pip install https://github.com/vllm-project/vllm/releases/download/v0.11.0/vllm-0.11.0+cpu-cp38-abi3-manylinux_2_35_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cpu