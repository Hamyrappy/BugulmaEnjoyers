import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from bugulma_enjoyers.models._base import BaseModel


class HFModel(BaseModel, model_type="hf"):
    def __init__(self, model_name: str, pipeline_config: dict, **kwargs: dict) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=torch.float16 if pipeline_config.device == "cuda" else torch.float32,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = pipeline_config

    def forward(self, batch: dict) -> str:
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            early_stopping=True,
            forced_bos_token_id=batch.get("forced_bos_token_id"),
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def to(self, device: torch.device) -> None:
        self.model.to(device)
        return self
