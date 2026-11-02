

instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be two types: 
(1) Search[query], which searches for a relevant question or topic on Wikipedia and returns the most relevant paragraphs.
(2) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""

def webthink(idx=None, prompt=None, to_print=True, max_steps=9):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    step_stats = []  # Step별 통계 수집
    
    for i in range(1, max_steps + 1):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        
        # Robust action parsing: take the first line and truncate after ] to handle hallucinations
        action = action.split('\n')[0].strip()
        if ']' in action:
            action = action[:action.find(']')+1]
        
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        
        # Step별 독성 문서 검색 통계 수집
        # action이 "search"로 시작하는 경우에만 독성 문서 검색이 발생
        action_lower = action.lower()
        is_search_action = action_lower.startswith('search[')
        
        if is_search_action:
            step_any_poisoned = info.get('is_poisoned', False)
            step_poisoned_count = info.get('poisoned_count', 0)
            step_total_count = info.get('total_count', 0)
            step_poisoned_flags = info.get('poisoned_flags', [])
        else:
            step_any_poisoned = False
            step_poisoned_count = 0
            step_total_count = 0
            step_poisoned_flags = []
        
        step_stats.append({
            'step': i,
            'thought': thought,
            'action': action,
            'observation': obs,
            'any_poisoned': step_any_poisoned,
            'poisoned_count': step_poisoned_count,
            'total_count': step_total_count,
            'is_search': is_search_action,
            'poisoned_flags': step_poisoned_flags
        })
        
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    
    if not done:
        obs, r, done, info = step(env, "finish[]")
        # finish action은 검색이 아니므로 통계에 포함하지 않음
    
    if to_print:
        print(info, '\n')
    info.update({
        'n_calls': n_calls, 
        'n_badcalls': n_badcalls, 
        'traj': prompt,
        'step_stats': step_stats  # Step별 통계 추가
    })
    return r, info
