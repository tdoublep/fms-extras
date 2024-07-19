#!/bin/bash

MODEL_ARGS="\
--variant=llama3.70b
--model_path="/gpfs/llama3/hf/70b_instruction_tuned"
--tokenizer="/gpfs/llama3/hf/70b_instruction_tuned"
--model_source=hf
--architecture=llama
--prompt_type="code"
--speculator_path="/gpfs/suneja/checkpoints/llama3-70b-ropefixed-tie_wt-scalednorm-4node-backup/checkpoints/step_11186_ckp.pth"
--speculator_ckpt_singlefile
--top_k_tokens_per_head=4,3,2,2
--batch_input
"

MODEL_ARGS_GRANITE_20B="\
--variant=ibm.20b
--model_path="/gpfs/prangan/granite-20b-code-instruct"
--tokenizer="/gpfs/prangan/granite-20b-code-instruct"
--model_source=hf
--architecture=gpt_bigcode
--prompt_type="code"
--speculator_path="/gpfs/suneja/checkpoints/grantite-20b-code-instruct-v1-speculator/step_42001_ckp.pth"
--speculator_load_type=singlefile
--top_k_tokens_per_head=4,3,2,2
"

MODEL_ARGS_GRANITE_20B_LOCAL="\
--variant=ibm.20b
--model_path="/gpfs/prangan/granite-20b-code-instruct"
--tokenizer="/gpfs/prangan/granite-20b-code-instruct"
--model_source=hf
--architecture=gpt_bigcode
--prompt_type="code"
--speculator_path="/gpfs/suneja/models/hub/models--ibm-granite--granite-20b-code-instruct-accelerator/snapshots/ba7a42036002f21f7536df750c398966b0cb8ad9"
--speculator_variant=1_7b
--speculator_source=hf
--speculator_load_type=registered_local
--top_k_tokens_per_head=4,3,2,2
"

MODEL_ARGS_GRANITE_20B_HF="\
--variant=ibm.20b
--model_path="/gpfs/prangan/granite-20b-code-instruct"
--tokenizer="/gpfs/prangan/granite-20b-code-instruct"
--model_source=hf
--architecture=gpt_bigcode
--prompt_type="code"
--speculator_path="/gpfs/suneja/models/granite-20b-code-instruct-accelerator"
--speculator_source=hf
--speculator_variant=1_7b
--speculator_load_type=hf_remote
--top_k_tokens_per_head=4,3,2,2
"
#--speculator_path="/gpfs/suneja/models/hub/models--ibm-granite--granite-20b-code-instruct-accelerator/snapshots/ba7a42036002f21f7536df750c398966b0cb8ad9"

MODEL_ARGS1="\
--variant="llama3.8b"
--model_path="/gpfs/llama3/hf/8b_instruction_tuned"
--tokenizer="/gpfs/llama3/hf/8b_instruction_tuned"
--model_source=hf
--architecture=llama
--speculator_path="/gpfs/suneja/models/hub/models--ibm-fms--llama3-8b-accelerator/snapshots/56660560945bc17b7958655bf15546bf8b11fced/"
--speculator_variant=3_2b
--speculator_source=hf
--top_k_tokens_per_head=4,3,2,2
"

MODEL_ARGS_GRANITE8B="\
--variant=calico.8b.code
--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-8b-code-instruct/snapshots/8a0fc76e4d374188e0cc8794d2d7275aa5aa7e64"
--tokenizer="/gpfs/prangan/hub/models--ibm-granite--granite-8b-code-instruct/snapshots/8a0fc76e4d374188e0cc8794d2d7275aa5aa7e64"
--model_source=hf
--architecture=llama
--prompt_type="code"
--speculator_ckpt_singlefile
--top_k_tokens_per_head=5,4,3,2,2
"
#--speculator_path="/gpfs/prangan/backup-ckptx/checkpoints/step_15001_ckp.pth"

MODEL_ARGS_GRANITE3B="\
--variant=calico.3b.code
--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-3b-code-instruct/snapshots/4420bfb5a3361ab4714bbd653848ef1a819d9f5b/"
--tokenizer="/gpfs/prangan/hub/models--ibm-granite--granite-3b-code-instruct/snapshots/4420bfb5a3361ab4714bbd653848ef1a819d9f5b/"
--model_source=hf
--architecture=llama
--prompt_type="code"
--speculator_ckpt_singlefile
--top_k_tokens_per_head=5,4,3,2,2
"
#--speculator_path="/gpfs/prangan/ckpts/granite_3b_stage1/checkpoints/step_21001_ckp.pth"

MODEL_ARGS_GRANITE3B_HF="\
--variant="calico.3b.code"
--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-3b-code-instruct/snapshots/4420bfb5a3361ab4714bbd653848ef1a819d9f5b/"
--tokenizer="/gpfs/prangan/hub/models--ibm-granite--granite-3b-code-instruct/snapshots/4420bfb5a3361ab4714bbd653848ef1a819d9f5b/"
--model_source=hf
--speculator_source=hf
--architecture=llama
--prompt_type="code"
--speculator_path="/gpfs/suneja/checkpoints/granite-3b/checkpoints/granite-3b-code-instruct/accelerator"
--speculator_variant=430m
--speculator_source=hf
--top_k_tokens_per_head=6,5,4,3,3
"

MODEL_ARGS_GRANITE34B_LOCAL="\
--variant="ibm.34b"
--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"
--tokenizer="/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"
--model_source=hf
--architecture=gpt_bigcode
--prompt_type="code"
--speculator_path="/gpfs/suneja/checkpoints/granite-34b-tp/checkpoints/granite-34b-code-instruct/accelerator"
--speculator_variant=680m
--speculator_source=hf
--speculator_load_type=registered_local
--top_k_tokens_per_head=6,5,4,3,3
"
#--speculator_path="/gpfs/suneja/checkpoints/granite-34b-tp/checkpoints/hf_ckpt_without_tie_wts/granite-34b-code-instruct/accelerator"

MODEL_ARGS_GRANITE34B_HF="\
--variant="ibm.34b"
--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"
--tokenizer="/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"
--model_source=hf
--architecture=gpt_bigcode
--prompt_type="code"
--speculator_variant=680m
--speculator_path="/gpfs/suneja/checkpoints/granite-34b-tp/checkpoints/granite-34b-code-instruct/accelerator"
--speculator_source=hf
--speculator_load_type=hf_remote
--top_k_tokens_per_head=6,5,4,3,3
"
#--speculator_path="/gpfs/suneja/checkpoints/granite-34b-tp/checkpoints/hf_ckpt_without_tie_wts/granite-34b-code-instruct/accelerator"
#--speculator_path="/gpfs/suneja/checkpoints/granite-34b-tp/checkpoints/hf_ckpt_with_tie_wts_v1/granite-34b-code-instruct/accelerator"

MODEL_ARGS_GRANITE34B="\
--variant="ibm.34b"
--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"
--tokenizer="/gpfs/prangan/hub/models--ibm-granite--granite-34b-code-instruct/snapshots/20f67e1f9b6016f62652916d7e887c7250c46382/"
--model_source=hf
--speculator_load_type=singlefile
--architecture=gpt_bigcode
--prompt_type="code"
--speculator_path="/gpfs/suneja/checkpoints/granite-34b-tp/checkpoints/step_21001_ckp.pth"
--top_k_tokens_per_head=6,5,4,3,3
"

MODEL_ARGS_CODELLAMA34B="\
--variant="34b.code"
--model_path="/gpfs/suneja/models/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d4c1c474abcacd32d2a6eda45f9811d38c83e93d"
--tokenizer="/gpfs/suneja/models/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d4c1c474abcacd32d2a6eda45f9811d38c83e93d"
--model_source=hf
--speculator_load_type=singlefile
--architecture=llama
--prompt_type="code"
--speculator_path="/gpfs/suneja/checkpoints/codellama-34b/checkpoints/step_21001_ckp.pth"
--top_k_tokens_per_head=6,5,4,3,3
"

MODEL_ARGS_CODELLAMA34B_LOCAL="\
--variant="34b.code"
--model_path="/gpfs/suneja/models/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d4c1c474abcacd32d2a6eda45f9811d38c83e93d"
--tokenizer="/gpfs/suneja/models/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d4c1c474abcacd32d2a6eda45f9811d38c83e93d"
--model_source=hf
--architecture=llama
--prompt_type="code"
--speculator_path="/gpfs/suneja/checkpoints/codellama-34b/checkpoints/CodeLlama-34b-Instruct-hf/accelerator/"
--speculator_variant=658m
--speculator_load_type=registered_local
--speculator_source=hf
--top_k_tokens_per_head=6,5,4,3,3
"

MODEL_ARGS_CODELLAMA34B_HF="\
--variant="34b.code"
--model_path="/gpfs/suneja/models/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d4c1c474abcacd32d2a6eda45f9811d38c83e93d"
--tokenizer="/gpfs/suneja/models/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d4c1c474abcacd32d2a6eda45f9811d38c83e93d"
--model_source=hf
--architecture=llama
--prompt_type="code"
--speculator_path="/gpfs/suneja/checkpoints/codellama-34b/checkpoints/CodeLlama-34b-Instruct-hf/accelerator/"
--speculator_variant=658m
--speculator_load_type=hf_remote
--speculator_source=hf
--top_k_tokens_per_head=6,5,4,3,3
"

MODEL_ARGS_LLAMA3_70B="\
--variant="llama3.70b"
--model_path="/gpfs/llama3/hf/70b_instruction_tuned"
--tokenizer="/gpfs/llama3/hf/70b_instruction_tuned"
--model_source=hf
--architecture=llama
--speculator_path="/gpfs/suneja/checkpoints/llama3-70b-ropefixed-tie_wt-scalednorm-4node-backup/checkpoints/step_11186_ckp.pth"
--speculator_load_type=singlefile
--speculator_variant=961m
--speculator_source=hf
--top_k_tokens_per_head=4,3,2,2
"

MODEL_ARGS_LLAMA3_70B_HF="\
--variant="llama3.70b"
--model_path="/gpfs/llama3/hf/70b_instruction_tuned"
--tokenizer="/gpfs/llama3/hf/70b_instruction_tuned"
--model_source=hf
--architecture=llama
--speculator_path="/gpfs/suneja/checkpoints/llama3-70b-ropefixed-tie_wt-scalednorm-4node-backup/checkpoints/Meta-Llama-3-70B-Instruct/accelerator"
--speculator_variant=961m
--speculator_source=hf
--speculator_load_type=hf_remote
--top_k_tokens_per_head=4,3,2,2
"

MODEL_ARGS_GRANITE13B_HF="\
--variant="ibm.13b"
--model_path="/gpfs/suneja/models/dmf_models/granite.13b.chat.v2.1-main/"
--tokenizer="/gpfs/suneja/models/dmf_models/granite.13b.chat.v2.1-main/"
--model_source=hf
--architecture=gpt_bigcode
--speculator_path="/gpfs/suneja/checkpoints/granite-13b-chat-v2.1-cconly/checkpoints/accelerator"
--speculator_source=hf
--speculator_load_type=hf_remote
--top_k_tokens_per_head=4,3,2,2
"

MODEL_ARGS_LLAMA2_70B_HF="\
--variant="70b"
--model_path="/gpfs/suneja/models/Llama-2-70b-chat-hf"
--tokenizer="/gpfs/suneja/models/Llama-2-70b-chat-hf"
--model_source=hf
--architecture=llama
--speculator_path="/gpfs/suneja/checkpoints/llama2-70b-tp-wtinitfix/checkpoints/Llama-2-70b-chat-hf/accelerator"
--speculator_source=hf
--speculator_load_type=hf_remote
--top_k_tokens_per_head=4,3,2,2
"


export CUDA_VISIBLE_DEVICES=1
torchrun \
    --nproc_per_node=1 \
    scripts/paged_speculative_inference.py \
    ${MODEL_ARGS_GRANITE13B_HF}