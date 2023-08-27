from flask import Flask, request, jsonify
import torch
from llama import Llama

app = Flask(__name__)

ckpt_dir =  r"D://codellama/CodeLlama-7b/"
tokenizer_path = r"D://codellama/CodeLlama-7b/tokenizer.model"
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=192,
    max_batch_size=4
)

@app.route('/generate', methods=['POST'])
def generate_prompt():
    try:
        data = request.json
        prompts = data.get('prompts', [])
        temperature = data.get('temperature', 0.2)
        top_p = data.get('top_p', 0.9)
        max_gen_len = data.get('max_gen_len', None)

        results = generator.text_completion(
            prompts=prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p
        )

        response_data = []
        for prompt, result in zip(prompts, results):
            response_data.append({
                'prompt': prompt,
                'result': result
            })

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/infill', methods=['POST'])
def infill_prompt():
    try:
        data = request.json
        prefixes = data.get('prefixes', [])
        suffixes = data.get('suffixes', [])
        temperature = data.get('temperature', 0.0)
        top_p = data.get('top_p', 0.9)
        max_gen_len = data.get('max_gen_len', 128)

        results = generator.text_infilling(
            prefixes=prefixes,
            suffixes=suffixes,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p
        )

        response_data = []
        for prefix, suffix, result in zip(prefixes, suffixes, results):
            response_data.append({
                'prefix': prefix,
                'suffix': suffix,
                'result': result
            })

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/chat', methods=['POST'])
def chat_completion():
    try:
        data = request.json
        dialogs = data.get('dialogs', [])
        temperature = data.get('temperature', 0.6)
        top_p = data.get('top_p', 0.9)
        max_gen_len = data.get('max_gen_len', None)

        results = generator.chat_completion(
            dialogs=dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p
        )

        response_data = []
        for dialog, result in zip(dialogs, results):
            response_data.append({
                'dialog': dialog,
                'result': result
            })

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
