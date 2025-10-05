from models.base_model import BaseModel
import torch
import transformers
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import os

class Llama3(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        using_distributed = os.environ.get("LOCAL_RANK", "-1") != "-1" or int(os.environ.get("WORLD_SIZE", "1")) > 1
        if using_distributed:
            device_map = None
        else:
            if torch.cuda.is_available():
                device_map = getattr(self.config, "device_map", {"": 0})
            else:
                device_map = getattr(self.config, "device_map", "auto")
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": question,
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": ans,
        }
        # Prefer explicit device pinning to avoid Accelerate "Device 0 unavailable" issues under multiprocessing
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.config.model_id,
            model_kwargs={"dtype": torch.bfloat16},
            device_map=device_map,
        )
        self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id
        self.pipeline.tokenizer.padding_side = "left"
        if self.config.get("model_path", None) is not None:
            print(f"Loading model from {self.config.model_path}")
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                dtype=torch.bfloat16,
                device_map=device_map,
            )
            self.pipeline.model = self.model
        else:
            self.model = self.pipeline.model
            
        if self.config.get("lora_path", None) is not None:
            print(f"Loading LoRA from {self.config.lora_path}")
            import peft
            self.model = peft.PeftModel.from_pretrained(
                self.model,
                self.config.lora_path,
                dtype=torch.bfloat16,
                device_map=device_map,
            )
            self.pipeline.model = self.model

    
    def create_text_message(self, texts, question): 
        prompt = ""
        for text in texts:
            prompt = prompt + text + '\n'
        message = {
            "role": "user",
            "content": f"{prompt}\n{question}",
        }
        return message
    
    @torch.no_grad()
    def predict(self, question, texts = None, images = None, history = None):
        self.clean_up()
        if images:
            print(f"Images are not supported for {self.config.model_id}")
            images = None
        messages = self.process_message(question, texts, images, history)
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id = self.pipeline.tokenizer.eos_token_id
        )
        self.clean_up()
        return outputs[0]["generated_text"][-1]['content'], outputs[0]["generated_text"]
        
    @torch.no_grad()
    def predict_batch(self, question_batch, text_batch = None, image_batch = None, history_batch = None):
        messages = [
            self.process_message(
                question=question,
                texts=text_batch[i] if text_batch is not None else None,
                images=image_batch[i] if image_batch is not None else None,
                history=history_batch[i] if history_batch is not None else None
            )
            for i, question in enumerate(question_batch)
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.config.max_new_tokens,
            batch_size=len(messages),
            pad_token_id = self.pipeline.tokenizer.eos_token_id
        )
        self.clean_up()
        results = []
        new_messages = []
        for gen in outputs:
            results.append(gen[0]['generated_text'][-1]['content'])
            new_messages.append(gen[0]['generated_text'])
        
        return results, new_messages
  
    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], str):
                return False
        return True

    def clean_up(self):
        torch.cuda.empty_cache()

class Llama4(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_id = config.model_id
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        if torch.cuda.is_available():
            device_map = {"": 0}
        else:
            device_map = "auto"
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_id,
            attn_implementation="flex_attention",
            device_map=device_map,
            dtype=torch.bfloat16,
        )

    def predict_message(self, messages):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
        )

        response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
        return response, None
    
    def clean_up(self):
        torch.cuda.empty_cache()
