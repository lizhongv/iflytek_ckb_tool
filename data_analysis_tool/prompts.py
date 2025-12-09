# -*- coding: utf-8 -*-

"""
Data analysis tool prompt templates
"""

# Problem normativity analysis prompt
PROBLEM_NORMATIVITY_PROMPT = """作为一位专业的问题规范性标注员，负责对输入问题进行规范性审核与分类标注。

## 判断标准与优先级

### 第一级：违规内容判定

如果问题包含以下内容，直接判定为违规：
- 违法信息：犯罪方法、违禁品制作、诈骗手段
- 安全威胁：恐怖主义、分裂国家、泄露机密
- 恶意攻击：种族/性别歧视、人身攻击、骚扰
- 违法违规：赌博、毒品、色情内容
- 隐私侵犯：索取身份证、电话号码等敏感信息
- 敏感政治：恶意歪曲历史、攻击政治制度

### 第二级：无意义内容判定

如果问题符合以下特征，判定为无意义：
- 纯符号堆砌：???、......、@@@@
- 随机字符组合：asdfghjkl、123456789
- 纯语气词重复：啊啊啊啊、嗯嗯嗯嗯
- 测试文本：test、测试、hello world
- 内容过短：≤2中文字符且无明确意义

### 第三级：模糊表述判定

如果问题同时满足以下两点，判定为模糊：
1. 包含未定义代词（这个/那个/他/她/它）
2. 无上下文支持，无法推断指代对象
注意：如"Python是什么？它有什么用途？"第二句有效

## 特殊场景处理

- 多语言混合但主题明确 → 规范
- 是否违规存疑 → 优先判无意义而非违规
- 多个缺陷 → 按优先级判定最高级别

## 输出格式要求

请严格遵循以下JSON格式输出判定结果：

```json
{{
  "is_standard": 0或1,
  "normativity_category": "Effective" | "Gibberish" | "Violation" | "Vague",
  "reason": "详细说明判断理由"
}}
```

## 字段说明

- `is_standard`: 0表示不规范问题，1表示规范问题
- `normativity_category`: 
  - 当 is_standard 为 1 时，必须填写 "Effective"
  - 当 is_standard 为 0 时，填写 "Gibberish"（无意义）、"Violation"（违规）或 "Vague"（模糊）中的一种
-`reason`：简要说明判定依据，包括问题核心与规范性标准的匹配或排除分析

用户问题：{question}"""

# Problem in/out set judgment prompt
PROBLEM_IN_OUT_SET_PROMPT = """作为{scenario}领域的业务专家，你需要判断用户问题是否属于{scenario}领域。

## 判定标准

### 集内问题（is_in_set=1）
如果问题的核心内容与{scenario}领域相关，特别是涉及{business_types}相关的政策、业务、流程或知识，则判定为集内问题。

### 集外问题（is_in_set=0）
如果问题的核心内容与{scenario}领域无关，属于其他专业领域（如医疗、法律、娱乐、技术编程、通用咨询等），则判定为集外问题。

## 判定原则
- **简单直接**：问题与{scenario}领域相关即为集内，无关即为集外
- **关注核心**：重点关注问题的核心诉求，而非仅看是否提及相关词汇
- **业务匹配**：如果问题涉及{business_types}中的任何业务类型，通常判定为集内

## 输出要求

请严格遵循以下JSON格式输出判定结果：

```json
{{
  "is_in_set": 0或1,
  "in_out_type": "具体的业务类型分类",
  "reason": "简要说明判断理由"
}}
```

## 字段说明
- `is_in_set`：整数，0表示集外问题（领域无关），1表示集内问题（领域相关）
- `in_out_type`：字符串
  - 当 `is_in_set` 为 1 时，必须填写{business_types}中列举的具体业务类型
  - 当 `is_in_set` 为 0 时，必须填写"out_of_domain"
- `reason`：简要说明判定依据

## 用户问题

{question}

请根据以上准则输出对用户问题判定的JSON格式结果。"""

# Recall judgment prompt (compare retrieved sources with correct source)
RECALL_JUDGMENT_PROMPT = """你是一位检索质量评估专家，专注于分析检索结果的准确性、全面性和一致性。
## 核心任务
基于用户问题、正确参考知识集合和模型检索返回的知识集合，判断检索结果是否完整、正确，并归类到七类检索问题类型中。

## 判断流程
1. 比较模型检索知识集合与正确参考知识集合；
2. 确认是否完整覆盖、语义一致、无冲突；
3. 根据判定规则呼出类别及理由。

## 注意
- 正确参考知识集合是回答用户问题所需的知识块集合；
- 模型检索知识集合是模型实际检索到的知识块；
- 需评估检索知识是否与正确知识含义一致，是否覆盖全部必需知识。

## 检索问题类型（七类，必须严格使用英文类型名称）：
1. **NoRecall**（溯源完全未召回）：模型未返回任何相关知识点，或返回的知识与正确知识无匹配。
2. **IncompleteRecall**（溯源召回不全面）：模型返回了部分相关正确的知识，但未全部召回全部的正确知识。
3. **MultiIntentIncomplete**（多意图召回不全）：用户问题包含多个意图/子问题，模型仅响应其中一部分。
4. **ComparisonIncomplete**（对比问题召回不全）：针对对比类问题，模型未全面覆盖比较维度或遗漏比较对象。
5. **TerminologyMismatch**（专业名词/口语化召回错误）：因术语与口语表达差异导致检索错误或误解。
6. **KnowledgeConflict**（检索知识冲突）：模型返回的知识块之间存在矛盾或冲突。
7. **CorrectRecall**（召回正确）：模型返回的知识完全覆盖正确参考知识，无错误或遗漏。

## 输出要求
- 若检索完全正确（即类型7 CorrectRecall），则 is_retrieval_correct 设为 1，否则设为 0
- retrieval_judgment_type **必须**填写上述七种英文类型之一：NoRecall、IncompleteRecall、MultiIntentIncomplete、ComparisonIncomplete、TerminologyMismatch、KnowledgeConflict、CorrectRecall
- retrieval_reason 提供简洁说明（≤50字），基于问题、正确知识与检索知识的对比分析

## 输出格式
请严格按照以下JSON格式输出，不得添加额外内容：

{{
  "is_retrieval_correct": 0或1,
  "retrieval_judgment_type": "上述七类检索类型之一",
  "retrieval_reason": "xxx（具体说明，最多50字）"
}}

## 输入数据

- 用户问题：{question}
- 正确参考知识集合：{correct_knowledge}
- 模型检索知识集合：{retrieved_source}

请严格按照上述规范对召回溯源文本与真实溯源的相关性进行判定，并以JSON格式输出结果。"""

# Recall judgment prompt (compare retrieved sources with correct answer)
RECALL_JUDGMENT_BY_ANSWER_PROMPT = """你是一位检索质量评估专家，专注于分析检索结果的准确性、全面性和一致性。
## 核心任务
基于用户问题、正确答案和模型检索返回的知识集合，判断检索结果是否完整、正确，并归类到七类检索问题类型中。

## 判断流程
1. 比较模型检索知识集合与正确答案；
2. 确认是否完整覆盖、语义一致、无冲突；
3. 根据判定规则呼出类别及理由。

## 注意
- 正确答案是回答用户问题的正确知识；
- 模型检索知识集合是模型实际检索到的知识块；
- 需评估检索知识是否与正确答案含义一致，是否覆盖全部必需知识。

## 检索问题类型（七类，必须严格使用英文类型名称）：
1. **NoRecall**（溯源完全未召回）：模型未返回任何相关知识点，或返回的知识与正确答案无匹配。
2. **IncompleteRecall**（溯源召回不全面）：模型返回了部分相关正确的知识，但未全部召回全部的正确知识。
3. **MultiIntentIncomplete**（多意图召回不全）：用户问题包含多个意图/子问题，模型仅响应其中一部分。
4. **ComparisonIncomplete**（对比问题召回不全）：针对对比类问题，模型未全面覆盖比较维度或遗漏比较对象。
5. **TerminologyMismatch**（专业名词/口语化召回错误）：因术语与口语表达差异导致检索错误或误解。
6. **KnowledgeConflict**（检索知识冲突）：模型返回的知识块之间存在矛盾或冲突。
7. **CorrectRecall**（召回正确）：模型返回的知识完全覆盖正确答案，无错误或遗漏。

## 输出要求
- 若检索完全正确（即类型7 CorrectRecall），则 is_retrieval_correct 设为 1，否则设为 0
- retrieval_judgment_type **必须**填写上述七种英文类型之一：NoRecall、IncompleteRecall、MultiIntentIncomplete、ComparisonIncomplete、TerminologyMismatch、KnowledgeConflict、CorrectRecall
- retrieval_reason 提供简洁说明（≤50字），基于问题、正确答案与检索知识的对比分析

## 输出格式
请严格按照以下JSON格式输出，不得添加额外内容：

{{
  "is_retrieval_correct": 0或1,
  "retrieval_judgment_type": "上述七类检索类型之一",
  "retrieval_reason": "xxx（具体说明，最多50字）"
}}

## 输入数据

- 用户问题：{question}
- 正确答案：{correct_answer}
- 模型检索知识集合：{retrieved_source}

请严格按照上述规范对召回溯源文本与正确答案的相关性进行判定，并以JSON格式输出结果。"""

# Response accuracy judgment prompt (compare model response with correct answer)
RESPONSE_ACCURACY_PROMPT = """你是一位回复质量评估专家，专注于分析模型回复的准确性、完整性和相关性。
## 核心任务
基于用户问题、正确答案和模型回复，判断回复是否正确，并归类到七类回复问题类型中。

## 判断流程
1. 比较模型回复与正确答案；
2. 确认是否完整覆盖、语义一致、无错误信息；
3. 根据判定规则呼出类别及理由。

## 注意
- 正确答案是回答用户问题的正确知识；
- 模型回复是模型实际生成的回答内容；
- 需评估回复是否与正确答案含义一致，是否完整覆盖全部必需信息。

## 回复问题类型（七类，必须严格使用英文类型名称）：
1. **Fully Correct**（完全正确）：模型回复完全符合正确答案，信息完整、准确、无遗漏。
2. **Partially Correct**（部分正确）：模型回复包含部分正确信息，但存在部分错误或不准确的内容。
3. **Incomplete Information**（信息不完整）：模型回复信息正确但不够完整，遗漏了部分重要信息。
4. **Incorrect Information**（信息错误）：模型回复包含错误信息，与正确答案不符。
5. **Irrelevant Answer**（无关回答）：模型回复与问题无关，未回答用户问题。
6. **Format Error**（格式错误）：模型回复格式不正确，无法正常阅读或理解。
7. **Other**（其他问题）：不属于上述六类的其他问题。

## 输出要求
- 若回复完全正确（即类型1 Fully Correct），则 is_response_correct 设为 1，否则设为 0
- response_judgment_type **必须**填写上述七种英文类型之一：Fully Correct、Partially Correct、Incomplete Information、Incorrect Information、Irrelevant Answer、Format Error、Other
- response_reason 提供简洁说明（≤50字），基于问题、正确答案与模型回复的对比分析

## 输出格式
请严格按照以下JSON格式输出，不得添加额外内容：

{{
  "is_response_correct": 0或1,
  "response_judgment_type": "上述七类回复类型之一",
  "response_reason": "xxx（具体说明，最多50字）"
}}

## 输入数据

- 用户问题：{question}
- 正确答案：{correct_answer}
- 模型回复：{model_response}

请严格按照上述规范对模型回复与正确答案的相关性进行判定，并以JSON格式输出结果。"""

# Response accuracy judgment prompt (compare model response with correct source)
RESPONSE_ACCURACY_BY_SOURCE_PROMPT = """你是一位回复质量评估专家，专注于分析模型回复的准确性、完整性和相关性。
## 核心任务
基于用户问题、正确参考知识集合和模型回复，判断回复是否正确，并归类到七类回复问题类型中。

## 判断流程
1. 比较模型回复与正确参考知识集合；
2. 确认是否完整覆盖、语义一致、无错误信息；
3. 根据判定规则呼出类别及理由。

## 注意
- 正确参考知识集合是回答用户问题所需的知识块集合；
- 模型回复是模型实际生成的回答内容；
- 需评估回复是否与正确知识含义一致，是否完整覆盖全部必需信息。

## 回复问题类型（七类，必须严格使用英文类型名称）：
1. **Fully Correct**（完全正确）：模型回复完全符合正确参考知识，信息完整、准确、无遗漏。
2. **Partially Correct**（部分正确）：模型回复包含部分正确信息，但存在部分错误或不准确的内容。
3. **Incomplete Information**（信息不完整）：模型回复信息正确但不够完整，遗漏了部分重要信息。
4. **Incorrect Information**（信息错误）：模型回复包含错误信息，与正确参考知识不符。
5. **Irrelevant Answer**（无关回答）：模型回复与问题无关，未回答用户问题。
6. **Format Error**（格式错误）：模型回复格式不正确，无法正常阅读或理解。
7. **Other**（其他问题）：不属于上述六类的其他问题。

## 输出要求
- 若回复完全正确（即类型1 Fully Correct），则 is_response_correct 设为 1，否则设为 0
- response_judgment_type **必须**填写上述七种英文类型之一：Fully Correct、Partially Correct、Incomplete Information、Incorrect Information、Irrelevant Answer、Format Error、Other
- response_reason 提供简洁说明（≤50字），基于问题、正确参考知识与模型回复的对比分析

## 输出格式
请严格按照以下JSON格式输出，不得添加额外内容：

{{
  "is_response_correct": 0或1,
  "response_judgment_type": "上述七类回复类型之一",
  "response_reason": "xxx（具体说明，最多50字）"
}}

## 输入数据

- 用户问题：{question}
- 正确参考知识集合：{correct_source}
- 模型回复：{model_response}

请严格按照上述规范对模型回复与正确参考知识的相关性进行判定，并以JSON格式输出结果。"""

