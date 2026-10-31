#!/usr/bin/env python3
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX_PATH = os.path.join(HERE, '..', '..', 'thesis_docs', 'main.tex')

print(f"Reading LaTeX file: {TEX_PATH}")
with open(TEX_PATH, 'r') as f:
    content = f.read()

# Regular expression to find equation environments
# Supports equation, equation*, align, align*, multline, multline*
eq_pattern = re.compile(
    r'(\\begin\{(equation|align|multline)\*?\}.*?\\end\{\2\*?\})',
    re.DOTALL
)

def fix_punctuation(match):
    full_block = match.group(1)
    env_name = match.group(2)
    
    # Extract the math content and the end environment tag
    end_tag = f"\\end{{{env_name}}}"
    if f"\\end{{{env_name}*}}" in full_block:
        end_tag = f"\\end{{{env_name}*}}"
        
    math_content = full_block[:-len(end_tag)].rstrip()
    
    # Find what follows the equation block in the main text
    idx_end = match.end()
    following_text = content[idx_end:].strip()
    
    # Strip any LaTeX comments immediately following
    while following_text.startswith('%'):
        next_newline = following_text.find('\n')
        if next_newline == -1:
            following_text = ""
            break
        following_text = following_text[next_newline:].strip()
        
    # Check the first character of following text to determine punctuation
    if not following_text:
        punct = "."
    elif following_text.startswith('\\chapter') or following_text.startswith('\\section') or following_text.startswith('\\subsection'):
        punct = "."
    elif following_text[0].isupper() or following_text.startswith('\n\n') or following_text.startswith('\\par'):
        punct = "."
    else:
        punct = ","
        
    # Check if the math content already ends with punctuation
    # Remove trailing spaces/newlines
    clean_math = math_content.rstrip()
    if clean_math.endswith('.') or clean_math.endswith(',') or clean_math.endswith(';') or clean_math.endswith('.') or clean_math.endswith(','):
        # Already punctuated, keep it
        return full_block
        
    # Apply punctuation before the end tag
    print(f"Adding '{punct}' to equation ending in: ... {clean_math[-20:]}")
    return f"{clean_math}{punct}\n{end_tag}"

new_content = eq_pattern.sub(fix_punctuation, content)

with open(TEX_PATH, 'w') as f:
    f.write(new_content)

print("Done fixing equation punctuation.")
