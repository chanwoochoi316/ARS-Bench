import re

def extract_answer_fn(output, mode='qa', extract_answer=False):
    if extract_answer == False and mode not in ['infogen', 'summary', 'research']:
        if mode == 'qa':
            # [수정] QA 모드에서도 \text{} 등 LaTeX 패턴을 제거하도록 처리
            raw_output = output.strip()
            # \text{...} 패턴을 찾아 내부 내용만 추출
            text_pattern = r'\\text\{([^}]*)\}'
            matches = re.findall(text_pattern, raw_output)
            if matches:
                return matches[-1].strip() # 마지막 \text{} 안의 내용 반환
            return raw_output
        pred_answer_lines = output.replace("\n\n", "\n").strip().split('\n')
        pred_answer = '\n'.join(pred_answer_lines[-3:])
        return pred_answer
    extracted_text = ''
    if mode == 'codegen':
        pattern = r'```python\s*(.*?)\s*```'  # Extract the code between ```python and ```
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode in ['infogen', 'summary', 'research']:
        pattern_info = "**Final Information"
        if "</think>\n" in output:
            extracted_text = output.split("</think>\n")[-1].split("<|begin_click_link|>")[0].replace(pattern_info, "").strip(':**').strip('\n').strip("```").strip()  # 提取</think>后面的内容
            if mode == 'infogen':
                extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # 只保留前5行
        elif pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].split("<|begin_click_link|>")[0].strip('\n').strip(':**').strip("```").strip()  # 提取**Final Information**后面的内容
            if mode == 'infogen':
                extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # 只保留前5行
        else:
            # extracted_text = "No helpful information found."
            # extracted_text = '\n'.join(output.strip().replace("</think>\n", "").replace("\n\n", "\n").split('\n')[-5:])  # 若没提取到，只保留最后5行
            extracted_text = output.strip()
        if mode == 'research':
            extracted_text = extracted_text[:6000]
        else:
            # extracted_text = extracted_text[:2500] # 임시 수정
            extracted_text = extracted_text
    elif mode in ['math', 'choose', 'qa']:
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
        else:
            # Try to match **ANSWER: ...** pattern first
            pattern_answer_bold = r'\*\*ANSWER:\s*(.*?)\*\*'
            matches_bold = re.findall(pattern_answer_bold, output, re.DOTALL)
            if matches_bold:
                extracted_text = matches_bold[-1].strip()  # Take the last match and strip whitespace
            else:
                # Try to match ANSWER: ... (with or without **)
                pattern_answer = r'ANSWER:\s*(.*?)(?:\*\*|\n\n|$)'
                matches_answer = re.findall(pattern_answer, output, re.DOTALL)
                if matches_answer:
                    extracted_text = matches_answer[-1].strip('**').strip()
                else:
                    # Fallback: simple split
                    pattern = 'ANSWER:'
                    if pattern in output:
                        extracted_text = output.split(pattern)[-1].strip('**').strip()
                        # Remove everything after first \n\n or **
                        if '\n\n' in extracted_text:
                            extracted_text = extracted_text.split('\n\n')[0]
                        elif '**' in extracted_text and extracted_text.count('**') > 1:
                            extracted_text = extracted_text.split('**')[0]
        if mode in ['choose','qa']:
            inner_pattern = r'\\text\{(.*)\}'
            inner_matches = re.findall(inner_pattern, extracted_text)
            if inner_matches:
                extracted_text = inner_matches[-1]  # Take the last match
            extracted_text = extracted_text.strip("()")
    return extracted_text
