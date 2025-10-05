import os
from models.base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
        
class Qwen25(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        using_distributed = os.environ.get("LOCAL_RANK", "-1") != "-1" or int(os.environ.get("WORLD_SIZE", "1")) > 1
        if using_distributed:
            device_map = None
        else:
            device_map = getattr(self.config, "device_map", "auto")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, dtype="auto", device_map=device_map)
        if self.config.lora_path:
            self.model = PeftModel.from_pretrained(self.model, self.config.lora_path)
            self.model = self.model.merge_and_unload()
            print(f"LoRA weights from {self.config.lora_path} merged into the model.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": question
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": ans
        }
        
    def create_text_message(self, texts, question):
        content = ""
        for text in texts:
            content += text + "\n"
        content += question
        message = {
            "role": "user",
            "content": content
        }
        return message
    
    @torch.no_grad()
    def predict(self, question, texts = None, images = None, history = None):
        self.clean_up()
        if images:
            print(f"Images are not supported for {self.config.model_id}")
            images = None
        messages = self.process_message(question, texts, images, history)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.config.max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        messages.append(self.create_ans_message(response))
        self.clean_up()
        return response, messages

    def clean_up(self):
        torch.cuda.empty_cache()

class Qwen3(Qwen25):
    def predict(self, question, texts = None, images = None, history = None):
        return self._predict(question, texts, images, history)
    
    @torch.no_grad()
    def _predict(self, question, texts = None, images = None, history = None):
        self.clean_up()
        if images:
            print(f"Images are not supported for {self.config.model_id}")
            images = None
        messages = self.process_message(question, texts, images, history)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.config.max_new_tokens,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        messages.append(self.create_ans_message(response))
        self.clean_up()
        
        return response, messages
