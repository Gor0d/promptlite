"""
PromptLite — Dataset de Benchmark
Prompts reais verbosos para testar o otimizador.
Cobre diferentes domínios e técnicas de otimização.
"""

BENCHMARK_PROMPTS = [
    {
        "id": "healthcare_001",
        "domain": "healthcare",
        "category": "summarization",
        "description": "Resumo clínico verboso",
        "prompt": """Hello! I hope you're doing well. I would really appreciate it if you could help me with something important. 
I need you to please summarize the following clinical report for me. It's very important that you include all the key findings, 
the diagnosis, the recommended treatments, and the patient's current status. Please make sure the summary is clear, 
concise, and easy to understand for medical professionals. Also, it's crucial that you don't miss any important details 
and that the summary is accurate. Can you please do that for me? Here is the clinical report:

{clinical_report}

Thank you so much for your help! I really appreciate it."""
    },
    {
        "id": "code_001", 
        "domain": "engineering",
        "category": "code_review",
        "description": "Code review verboso",
        "prompt": """I would like you to act as an experienced senior software engineer with deep expertise in Python. 
As an AI assistant, you have the ability to analyze code and provide detailed, constructive feedback. 
I am going to share some Python code with you, and I would really appreciate it if you could review it thoroughly. 
Please look at things like code quality, potential bugs, performance issues, security vulnerabilities, and adherence 
to Python best practices and PEP 8 style guidelines. It would also be very helpful if you could suggest specific 
improvements and explain why each suggestion would make the code better. Please be thorough but also clear in your 
explanations so that a developer of intermediate experience level could understand your feedback. Here is the code:

{python_code}

I look forward to your detailed review and thank you in advance for your time and expertise."""
    },
    {
        "id": "data_001",
        "domain": "data_science", 
        "category": "analysis",
        "description": "Análise de dados verbosa",
        "prompt": """Please help me analyze this dataset. I need you to look at the data carefully and provide insights. 
It's important that you examine the data from multiple angles. First, please describe what you see in the data overall. 
Then, please identify any patterns or trends that might be present. After that, please point out any anomalies or 
outliers that could be significant. It would also be very useful if you could suggest what additional analysis might 
be valuable to perform. Please make sure your analysis is thorough and that you don't skip any important aspects. 
Also, please format your response in a clear and organized way so it's easy to follow. Here is the dataset:

{dataset}

Please provide a comprehensive analysis."""
    },
    {
        "id": "business_001",
        "domain": "business",
        "category": "email",
        "description": "Geração de email verbosa",
        "prompt": """I need your assistance in writing a professional business email. This is a very important email 
that needs to be written carefully and professionally. The email is going to be sent to a potential client who we 
are hoping to convert into an actual paying customer. We want to follow up on a meeting that we had with them 
last week where we discussed our software product. In the meeting, we talked about how our product could solve 
their specific business problems. We need the email to be polite, professional, and persuasive without being 
pushy or aggressive. It should remind them of the key benefits we discussed, address any concerns they might have, 
and include a clear call to action asking them to schedule a demo or a follow-up call. The tone should be warm 
but professional. Please write this email for me, making sure it's well-structured with a good subject line, 
a proper greeting, a clear body, and a professional sign-off."""
    },
    {
        "id": "llm_training_001",
        "domain": "ai_training",
        "category": "instruction_following",
        "description": "Instrução de treinamento de LLM verbosa",
        "prompt": """As an AI language model, I want you to understand that you should always try to be helpful, 
harmless, and honest in your responses. When answering questions, please make sure that you provide accurate 
information to the best of your ability. If you are not sure about something, please say so rather than 
making something up. It is very important that you do not provide false or misleading information. 
Additionally, please be respectful and considerate in all your responses, avoiding any content that could 
be offensive or harmful to users. Now, with all of that in mind, I would like you to explain the concept 
of machine learning to me in simple terms that someone without a technical background could understand. 
Please use analogies and examples to make the explanation as clear and accessible as possible."""
    },
    {
        "id": "rag_001",
        "domain": "ai_engineering",
        "category": "rag_prompt",
        "description": "System prompt RAG verboso",
        "prompt": """You are a helpful AI assistant that has been provided with relevant context documents to help 
answer user questions. Your primary role is to assist users by providing accurate and helpful responses based 
on the context that has been retrieved and provided to you. It is very important that you base your answers 
primarily on the provided context rather than relying solely on your training data. If the answer to a question 
can be found in the provided context, please use that information to formulate your response. If the context 
does not contain sufficient information to answer the question, please clearly indicate this to the user and 
let them know that the information is not available in the provided documents. Please do not make up information 
or hallucinate facts that are not supported by the context. Always cite which part of the context supports 
your answer when possible. Here is the retrieved context:

{context}

Please answer the following question based on the above context:
{question}"""
    }
]


def get_prompt_by_id(prompt_id: str) -> dict | None:
    return next((p for p in BENCHMARK_PROMPTS if p["id"] == prompt_id), None)


def get_prompts_by_domain(domain: str) -> list[dict]:
    return [p for p in BENCHMARK_PROMPTS if p["domain"] == domain]
