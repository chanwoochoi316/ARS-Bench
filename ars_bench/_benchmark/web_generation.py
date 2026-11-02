from __future__ import annotations
from .webthinker import END_SEARCH_QUERY

async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    generate_mode: str = "chat",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 32768,
    repetition_penalty: float = 1.0,
    top_k: int = 1,
    min_p: float = 0.0,
    model_name: str = "QwQ-32B",
    stop: List[str] = [END_SEARCH_QUERY],
    retry_limit: int = 3,
    bad_words: List[str] = None,
    tokenizer=None,
    aux_tokenizer=None,
    seed: int = None
) -> Tuple[str, str]:
    """Generate a single response with retry logic"""
    if bad_words is None:
        bad_words = [f"{END_SEARCH_RESULT}\n\n{tokenizer.eos_token if tokenizer else ''}"]
    for attempt in range(retry_limit):
        try:
            async with semaphore:
                if generate_mode == "chat":
                    messages = [{"role": "user", "content": prompt}]
                    # Determine which tokenizer/model to use logic is external or simple here
                    # For simplicity, if tokenizer provided use it
                    if tokenizer and ('qwq' in model_name.lower() or 'deepseek' in model_name.lower() or 'r1' in model_name.lower() or 'qwen' in model_name.lower()):
                        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        print(f"[DEBUG] Formatted Prompt: {formatted_prompt}")
                    elif aux_tokenizer:
                        formatted_prompt = aux_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        formatted_prompt = prompt # Fallback

                    if ('deepseek' in model_name.lower() or 'r1' in model_name.lower()) and "<think>\n" not in formatted_prompt:
                        formatted_prompt = formatted_prompt + "<think>\n"
                else:
                    formatted_prompt = prompt

                response = await client.completions.create(
                    model=model_name,
                    prompt=formatted_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    seed=seed,
                    extra_body={
                        'top_k': top_k,
                        'include_stop_str_in_output': True,
                        'repetition_penalty': repetition_penalty,
                        # 'bad_words': bad_words,
                        # 'min_p': min_p
                    },
                    timeout=3600,
                )

                text = response.choices[0].text
                if not text:
                    print(f"[DEBUG] Empty response! Finish Reason: {response.choices[0].finish_reason}")
                    print(f"[DEBUG] Prompt Tail (last 500): {formatted_prompt[-500:]!r}")

                return formatted_prompt, text
        except Exception as e:
            print(f"[Error] generate_response failed: {e}")
            if "maximum context length" in str(e).lower():
                max_tokens = max_tokens // 2
            if attempt == retry_limit - 1:
                return "", ""
            await asyncio.sleep(1 * (attempt + 1))
    return "", ""
