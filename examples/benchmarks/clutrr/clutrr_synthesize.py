import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.clutrr.resources import CLUTRRResource
from cartridges.utils.wandb import WandBConfig


client = TokasaurusClient.Config(
    url="http://0.0.0.0:10210",
    model_name="Qwen/Qwen3-4b",
)

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.75,
        use_tools_a=False,
        use_tools_b=False,
        tools=[],
        resources=[
            CLUTRRResource.Config(
                config="gen_train23_test2to10",
                split="train",
                noise_types=[1],        # clean stories only
                stories_per_prompt=5,
                seed_prompts=["question", "summarization", "generic"],
            )
        ],
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=32768,
    batch_size=32,
    max_num_batches_in_parallel=256,
    name=FormatStringVariable(f"{Path(__file__).stem}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(tags=["clutrr_synthesis"]),
    upload_to_wandb=False,
    save_wandb_preview=False,
)

if __name__ == "__main__":
    pydrantic.main([config])
