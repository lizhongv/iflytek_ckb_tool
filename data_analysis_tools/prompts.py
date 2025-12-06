# -*- coding: utf-8 -*-

"""
Data analysis tool prompt templates
"""

# Problem normativity analysis prompt
PROBLEM_NORMATIVITY_PROMPT = """你是一位专业的问题规范性标注员，负责对输入问题进行标准化审核与分类标注。

## 判断标准

### 一、是否规范问题（is_standard）
需同时满足以下三点：
1. **有实质内容**：包含明确的主题或信息需求，不是无意义的字符堆砌
2. **符合公序良俗**：不包含违法、违规、歧视、攻击性内容或敏感信息
3. **逻辑清晰可理解**：表述基本通顺，有明确的疑问指向

### 二、问题类型（question_type）
从以下四类中选择：
- **Effective**：有明确主题、需要实质解答的问题（规范问题必须为此类型）
- **Meaningless**：无实质内容、语气词堆砌、随机字符（如："啊啊啊？？"、"123abc"）
- **Violation**：包含违法信息、恶意言论、人身攻击、敏感政治内容等
- **Ambiguous**：表述笼统、主语指代不明，无法确定具体对象（如："这个怎么办？"、"那是什么？"）

## 标注规则
1. **优先级判断**：
   - 违规内容 → is_standard=0, question_type="Violation"
   - 无实质内容 → is_standard=0, question_type="Meaningless"
   - 指代不明 → is_standard=0, question_type="Ambiguous"
   - 其他规范问题 → is_standard=1, question_type="Effective"

2. **客观性原则**：只关注问题本身，不考虑回答难度或知识准确性

## 输出格式
严格按照以下JSON格式输出，字段名称不可修改：

```json
{{
  "is_standard": 0或1,
  "question_type": "Effective" | "Meaningless" | "Violation" | "Ambiguous",
  "reason": "简要说明判断依据"
}}
```

**注意**：
- 当is_standard=1时，question_type必须为"Effective"
- 当is_standard=0时，question_type为其他三种类型之一

输入问题：{question}"""

# Problem in/out set judgment prompt
PROBLEM_IN_OUT_SET_PROMPT = """你是一个{scenario}领域的业务专家，专精于{business_types}所涵盖的各项业务。现在需要对用户提问进行**精准的领域相关性判定**。

## 核心任务
判断用户问题是否属于您所负责的 {scenario} 领域，并确定其具体归属的业务类型。您需要：
1. 分析用户问题的核心意图、主题和所需知识
2. 判断问题是否属于 {scenario} 领域，并涉及 {business_types} 中的业务类型
3. 对问题进行分类：集内（属于领域）或集外（不属于领域）
4. 如果属于集内，明确标注对应的具体业务类型

## 判定标准
- **集内问题**：用户问题在主题、意图或所需知识上，明确属于您所负责的 {scenario} 领域,并具体涉及 {business_types} 中的至少一个业务类型。
- **集外问题**：用户问题与您的领域无直接关联，或其核心诉求属于其他专业知识范畴（如医疗、法律、娱乐、技术编程、通用咨询等）。

## 判定逻辑与流程

1. **问题解析**
- 提取用户问题 {question} 中的核心实体、关键词、意图动词。
- 分析问题的潜在场景和最终目标。

2. **领域匹配**
- 将解析出的关键词和意图，与您所负责的领域 {scenario} 和具体业务类型 {business_types} 进行比对。
- **重点匹配**：判断问题是否需要您所在领域的专业知识、流程、政策或数据才能解答。

3. **模糊处理**
- 如果问题处于模糊地带，例如提及了领域内的通用词汇但实际意图是其他领域（如"社保"在财务软件或小说情节中被提及），应判定为**集外**。
- 如果问题仅涉及非常浅层的背景信息，但核心是其他领域的操作（如"打印社保凭证的打印机如何设置"），应判定为**集外**。

## 输出要求：

请严格遵循以下JSON格式输出判定结果：

{{
  "is_in_set": 0或1,
  "in_out_type": "具体的业务类型分类",
  "reason": "详细说明判断理由"
}}

## 字段说明：
- `is_in_set`：0表示集外问题（领域无关），1表示集内问题（领域相关）。
- `in_out_type`：
  - 当 `is_in_set` 为 1 时，填写该问题所对应的具体业务类型（必须是 {business_types} 中列举的类型之一）。
  - 当 `is_in_set` 为 0 时，填写 "out_of_domain"。
- `reason`：简要说明判定依据，包括问题核心与领域/业务类型的匹配或排除分析。

请根据以上准则，对用户问题 {question} 进行判定。"""

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
RESPONSE_ACCURACY_PROMPT = """你是一个专业的回复质量评估专家。请分析以下模型回复是否正确。

问题：{question}
正确答案：{correct_answer}
模型回复：{model_response}

请按照以下JSON格式输出：

{{
  "is_response_correct": 0或1,
  "response_judgment_type": "七大类型之一：1.Fully Correct 2.Partially Correct 3.Incomplete Information 4.Incorrect Information 5.Irrelevant Answer 6.Format Error 7.Other",
  "response_reason": "详细说明判断理由"
}}
"""

# Response accuracy judgment prompt (compare model response with correct source)
RESPONSE_ACCURACY_BY_SOURCE_PROMPT = """你是一个专业的回复质量评估专家。请分析以下模型回复是否与正确溯源匹配。

问题：{question}
正确溯源：{correct_source}
模型回复：{model_response}

请按照以下JSON格式输出：

{{
  "is_response_correct": 0或1,
  "response_judgment_type": "七大类型之一：1.Fully Correct 2.Partially Correct 3.Incomplete Information 4.Incorrect Information 5.Irrelevant Answer 6.Format Error 7.Other",
  "response_reason": "详细说明判断理由"
}}
"""

