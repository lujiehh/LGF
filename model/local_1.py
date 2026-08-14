import os
import sys

# 将开源包根目录加入搜索路径，以便 import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
import re
import time
import datetime

import config

# 设置API端点和模型名称
API_ENDPOINT = config.LLM_API_ENDPOINT
MODEL_NAME = config.LLM_MODEL_NAME

def call_local_llm(prompt, system_prompt=None, max_tokens=1024, temperature=0.0, top_p=1.0):
    """
    调用本地Llama-3.1-8B-Instruct模型的API
    
    参数：
    prompt: 用户输入的提示词
    system_prompt: 系统提示词（可选）
    max_tokens: 生成的最大token数
    temperature: 生成温度
    top_p: 核采样参数
    
    返回：
    tuple: (生成的文本, 置信度)
    """
    # 构造请求头
    headers = {
        "Content-Type": "application/json"
    }
    
    # 构造请求体 - 添加logprobs参数以获取置信度信息
    payload = {
        "model": MODEL_NAME,
        "messages": [],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": True,  # 请求返回置信度信息
        "top_logprobs": 1  # 返回最高概率的token信息
    }
    
    # 添加系统提示词（如果有）
    if system_prompt:
        payload["messages"].append({
            "role": "system",
            "content": system_prompt
        })
    
    # 添加用户提示词
    payload["messages"].append({
        "role": "user",
        "content": prompt
    })
    
    try:
        # 发送请求
        response = requests.post(
            f"{API_ENDPOINT}/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
        
        # 检查响应状态
        if response.status_code == 200:
            # 解析响应
            result = response.json()
            
            # 获取生成的文本
            generated_text = result["choices"][0]["message"]["content"].strip()
            
            # 尝试获取置信度信息
            confidence = None
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "logprobs" in choice and choice["logprobs"] is not None:
                    # 计算整体置信度 - 取所有token概率的平均值
                    if "content" in choice["logprobs"] and len(choice["logprobs"]["content"]) > 0:
                        total_logprob = 0
                        for token_info in choice["logprobs"]["content"]:
                            if "logprob" in token_info:
                                total_logprob += token_info["logprob"]
                        avg_logprob = total_logprob / len(choice["logprobs"]["content"])
                        # 转换为概率值（0-1范围）
                        confidence = min(max(0, 10**avg_logprob), 1)
            
            return generated_text, confidence
        else:
            print(f"API请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None, None
    except Exception as e:
        print(f"调用API时发生错误: {str(e)}")
        return None, None

def get_entity_sentences(premise, head_entity, tail_entity):
    """
    获取包含头尾实体的句子
    
    参数：
    premise: 前提句子列表
    head_entity: 头实体
    tail_entity: 尾实体
    
    返回：
    list: 包含头尾实体的句子列表
    """
    entity_sentences = []
    seen_sentences = set()
    
    for sentence in premise:
        # 检查句子是否包含头实体或尾实体
        if (head_entity in sentence or tail_entity in sentence) and sentence not in seen_sentences:
            entity_sentences.append(sentence)
            seen_sentences.add(sentence)
    
    return entity_sentences

def extract_entities_from_task(task):
    """
    从任务中提取头尾实体
    
    参数：
    task: 任务字典
    
    返回：
    tuple: (头实体, 尾实体, 头实体类型, 尾实体类型)
    """
    head_entity = ""
    tail_entity = ""
    head_type = ""
    tail_type = ""
    
    # 从head_type_pair字段提取头实体和类型
    if "head_type_pair" in task and isinstance(task["head_type_pair"], list) and len(task["head_type_pair"]) >= 2:
        head_entity = task["head_type_pair"][0]
        head_type = task["head_type_pair"][1]
    
    # 从tail_type_pair字段提取尾实体和类型
    if "tail_type_pair" in task and isinstance(task["tail_type_pair"], list) and len(task["tail_type_pair"]) >= 2:
        tail_entity = task["tail_type_pair"][0]
        tail_type = task["tail_type_pair"][1]
    
    return head_entity, tail_entity, head_type, tail_type

def process_nli_data(input_file, output_file):
    """
    处理NLI数据，使用头尾实体所在的句子组合作为前提句
    
    参数：
    input_file: 输入JSON文件路径
    output_file: 输出JSON文件路径
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        nli_tasks = json.load(f)
    
    total_tasks = len(nli_tasks)
    print(f"总任务数: {total_tasks}")
    
    # 系统提示词
    system_prompt1 = "你是一个逻辑推理专家，擅长判断前提句是否能直接推断出假设句。"
    
    results = []
    
    for i, task in enumerate(nli_tasks):
        premise = task["premise"]
        hypothesis = task["hypothesis"]
        head_type = task.get("head_type", "")
        tail_type = task.get("tail_type", "")
        
        # 提取头尾实体和类型
        head_entity, tail_entity, extracted_head_type, extracted_tail_type = extract_entities_from_task(task)
        
        # 优先使用提取的类型，如果没有则使用原有的类型
        if extracted_head_type:
            head_type = extracted_head_type
        if extracted_tail_type:
            tail_type = extracted_tail_type
        
        # 输出当前处理的信息
        print(f"\n任务 {i+1}/{total_tasks}:")
        print(f"假设 (hypothesis): {hypothesis}")
        print(f"头实体: {head_entity}")
        print(f"尾实体: {tail_entity}")
        print(f"头实体类型: {head_type}")
        print(f"尾实体类型: {tail_type}")
        
        # 确保premise是列表
        if isinstance(premise, str):
            premise = [premise]
        
        print(f"原始前提句子数: {len(premise)}")
        
        # 获取包含头尾实体的句子
        entity_sentences = get_entity_sentences(premise, head_entity, tail_entity)
        print(f"包含头尾实体的句子数: {len(entity_sentences)}")
        
        # 如果没有找到包含头尾实体的句子，使用所有句子
        if not entity_sentences:
            entity_sentences = premise
            print("⚠️  未找到包含头尾实体的句子，使用所有句子")
        
        # 构建前提文本
        premise_text = ' '.join(entity_sentences)
        print(f"构建的前提文本: {premise_text}")
        
        # 构建提示词
        user_prompt1 = f"""
        请判断是否可以通过前提句直接推断出假设句。
        前提：{premise_text}
        假设：{hypothesis}
        
        请只输出true或false，不要添加任何其他解释：
        """
        
        # 调用LLM，获取判断结果
        response1, confidence1 = call_local_llm(user_prompt1, system_prompt=system_prompt1)
        
        # 解析判断的输出
        result1 = None
        if response1 is not None:
            # 忽略<think>...</think>标记内的思考内容
            cleaned_response1 = re.sub(r'<think>.*?(?:</think>|$)', '', response1, flags=re.DOTALL)
            
            # 添加打印语句，显示原始的LLM输出
            print(f"LLM清理输出: {repr(cleaned_response1)}")
            
            # 在剩余内容中查找true或false关键词
            if "true" in cleaned_response1.lower():
                result1 = True
            elif "false" in cleaned_response1.lower():
                result1 = False
            else:
                # 如果找不到明确的true/false，默认设为none
                result1 = None
        
        # 输出判断结果和置信度
        if result1 is not None:
            print(f"判断结果: {result1}")
            if confidence1 is not None:
                print(f"置信度: {confidence1:.4f}")
            else:
                print("置信度: 无法获取")
        else:
            print("判断结果: None (无法解析LLM输出)")
        
        # 确定最终答案
        final_answer = result1
        final_confidence = confidence1
        
        # 复制原始任务数据并添加answer和confidence字段
        result = task.copy()
        result["answer"] = final_answer
        result["confidence"] = final_confidence
        result["entity_based_premise"] = premise_text
        result["head_entity"] = head_entity
        result["tail_entity"] = tail_entity
        result["head_entity_type"] = head_type
        result["tail_entity_type"] = tail_type
        result["entity_sentences_count"] = len(entity_sentences)
        
        results.append(result)
        
        # 打印进度
        if (i + 1) % 10 == 0 or (i + 1) == total_tasks:
            print(f"\n处理进度：{i + 1}/{total_tasks}")
    
    # 保存结果到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nNLI任务处理完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    # 设置文件路径
    input_file = config.DATA_PATH
    output_file = config.LOCAL_RESULT_PATH
    
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.datetime.now()
    
    # 执行处理
    process_nli_data(input_file, output_file)
    
    # 记录结束时间
    end_time = time.time()
    end_datetime = datetime.datetime.now()
    
    # 计算运行时间
    run_time_seconds = end_time - start_time
    run_time_minutes = run_time_seconds / 60
    run_time_hours = run_time_minutes / 60
    
    # 构造日志信息
    log_message = f"""
====================================
运行时间日志 - run_local_llama_entity_based.py
====================================
开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}
结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}
总运行时间: {run_time_seconds:.2f} 秒 ({run_time_minutes:.2f} 分钟 / {run_time_hours:.2f} 小时)
输入文件: {input_file}
输出文件: {output_file}
模型名称: {MODEL_NAME}
====================================
"""
    
    # 输出到控制台
    print("\n" + log_message)
    
    # 写入日志文件
    log_file_path = os.path.join(os.path.dirname(__file__), "run_local_llama_entity_based.log")
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(log_message)
    
    print(f"日志已保存到: {log_file_path}")