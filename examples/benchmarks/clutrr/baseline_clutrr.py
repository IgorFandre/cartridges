import os

import pydrantic

from cartridges.clients.openai import CartridgeConfig, OpenAIClient
from cartridges.data.clutrr.resources import CLUTRRResource
from cartridges.evaluate import ICLBaseline, GenerationEvalRunConfig
from cartridges.data.clutrr.evals import CLUTRRRelationGenerateDataset
from cartridges.evaluate import GenerationEvalConfig
from cartridges.utils.wandb import WandBConfig


base_url = os.environ.get("CARTRIDGES_VLLM_QWEN3_4B_URL", "http://localhost:8000")

client = OpenAIClient.Config(
    base_url=os.path.join(base_url, "v1"),
    model_name="Qwen/Qwen3-4b",
)

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert at reasoning about family relationships. \
Use the kinship stories below to answer questions about family relations.

<kinship-stories>
{content}
</kinship-stories>

Think carefully about multi-hop relationships (e.g. mother's brother = uncle).
Keep your answer to a single relationship word inside <answer> tags.
"""

configs = [
    GenerationEvalRunConfig(
        name="clutrr_icl_baseline",
        generator=ICLBaseline.Config(
            client=client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.3,
            max_completion_tokens=512,
            context=CLUTRRResource.Config(
                config="gen_train23_test2to10",
                split="train",
                noise_types=[1],
            ),
        ),
        eval=GenerationEvalConfig(
            dataset=CLUTRRRelationGenerateDataset.Config(
                config="gen_train23_test2to10",
                split="test",
                noise_types=[1],
                cot=True,
            ),
            name_for_wandb="clutrr_test",
            num_samples=1,
            temperature=0.3,
        ),
        max_num_batches_in_parallel=32,
        batch_size=32,
        wandb=WandBConfig(tags=["clutrr", "genbaseline", "icl"]),
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
