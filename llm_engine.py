
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
import torch
from rich.console import Console
from config import LLM_MODEL, LLM_REVISION

console = Console()

class LocalLLM:
    def __init__(self):
        self.pipe = None
        self.load_time = None
        self._load_model()

    def _load_model(self):
        console.log(
            f"[cyan]Loading LLM: {LLM_MODEL}[/cyan]")
        console.log(
            "[yellow]This may take a few minutes "
            "on first run...[/yellow]")
        t0 = time.time()

        # Use CPU with float32 for compatibility
        device = "cuda" if torch.cuda.is_available() \
                 else "cpu"
        console.log(f"[cyan]Device: {device}[/cyan]")

        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL,
            revision=LLM_REVISION
        )

        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            revision=LLM_REVISION,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=300,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

        self.load_time = round(time.time() - t0, 2)
        console.log(
            f"[green]✅ LLM loaded in "
            f"{self.load_time}s[/green]")

    def generate(self, prompt: str) -> str:
        """Generate response from prompt"""
        t0 = time.time()

        # TinyLlama chat format
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful support agent "
                    "for OrbitDesk. Answer only from "
                    "the provided context. Be concise "
                    "and accurate."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Format as chat template
        tokenizer = self.pipe.tokenizer
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        output = self.pipe(formatted)
        raw = output[0]["generated_text"]

        # Extract only the assistant's response
        if "<|assistant|>" in raw:
            response = raw.split(
                "<|assistant|>")[-1].strip()
        else:
            # Fallback: remove the prompt
            response = raw[len(formatted):].strip()

        latency = round(time.time() - t0, 2)
        console.log(
            f"[dim]LLM latency: {latency}s[/dim]")

        return response