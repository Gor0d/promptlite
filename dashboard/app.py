"""
PromptLite — Dashboard Streamlit
"""

import streamlit as st
import requests
import json

st.set_page_config(page_title="PromptLite", page_icon="⚡", layout="wide")

API_URL = "http://localhost:8000"

st.markdown("""
<style>
.grade-A { color: #2e7d32; font-size: 2rem; font-weight: 700; }
.grade-B { color: #1565c0; font-size: 2rem; font-weight: 700; }
.grade-C { color: #e65100; font-size: 2rem; font-weight: 700; }
.grade-D { color: #b71c1c; font-size: 2rem; font-weight: 700; }
.metric-box { background: #f8f9fa; border-radius: 8px; padding: 1rem; text-align: center; }
.prompt-box { background: #f0f4ff; border-left: 4px solid #1565c0; border-radius: 4px; padding: 1rem; font-family: monospace; font-size: 13px; white-space: pre-wrap; }
.optimized-box { background: #f0fff4; border-left: 4px solid #2e7d32; border-radius: 4px; padding: 1rem; font-family: monospace; font-size: 13px; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

st.markdown("# ⚡ PromptLite")
st.markdown("##### Otimize prompts · Reduza tokens · Preserve intenção")

tab1, tab2, tab3 = st.tabs(["🔧 Otimizador", "📊 Benchmark", "📚 Técnicas"])

# ─── TAB 1 — OTIMIZADOR ───
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        examples = {
            "-- Cole seu prompt abaixo --": "",
            "Healthcare — resumo clínico verboso": "Hello! I hope you're doing well. I would really appreciate it if you could help me with something important. I need you to please summarize the following clinical report for me. It's very important that you include all the key findings, the diagnosis, and the recommended treatments. Please make sure the summary is clear and concise.",
            "Code review verboso": "I would like you to act as an experienced senior software engineer. I am going to share some Python code with you, and I would really appreciate it if you could review it thoroughly. Please look at code quality, potential bugs, performance issues, and adherence to best practices. Here is the code: {python_code}",
            "RAG system prompt verboso": "You are a helpful AI assistant that has been provided with relevant context documents to help answer user questions. It is very important that you base your answers primarily on the provided context. If the context does not contain sufficient information, please clearly indicate this to the user. Do not make up information. Here is the context: {context}. Please answer: {question}",
        }

        selected = st.selectbox("💡 Exemplos:", list(examples.keys()))
        prompt_input = st.text_area(
            "Seu prompt:",
            value=examples[selected],
            height=200,
            placeholder="Cole aqui um prompt verboso para otimizar..."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            test_outputs = st.checkbox("Testar equivalência de outputs", value=False,
                                       help="Executa o prompt original e otimizado no LLM e compara os resultados. Usa mais tokens.")
        with col_b:
            optimize_btn = st.button("⚡ Otimizar", type="primary", use_container_width=True)

    with col2:
        st.markdown("### ℹ️ Como funciona")
        st.info("""
**Pipeline de otimização:**

1. Conta tokens com `tiktoken`
2. Extrai intenção core com LLM
3. Remove redundâncias e verbosidade
4. Reescreve em forma imperativa
5. Testa equivalência de outputs
6. Calcula score e grade

**Grades:**
- **A** — excelente redução, intenção preservada
- **B** — boa redução, pequena perda
- **C** — redução moderada
- **D** — output mudou, não usar
        """)

    if optimize_btn and prompt_input.strip():
        with st.spinner("⚡ Otimizando prompt..."):
            try:
                resp = requests.post(
                    f"{API_URL}/optimize",
                    json={"prompt": prompt_input, "test_outputs": test_outputs},
                    timeout=60
                )

                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown("---")

                    # Grade + métricas principais
                    g1, g2, g3, g4, g5 = st.columns(5)
                    grade = data["grade"]
                    g1.markdown(f'<div class="metric-box"><div style="font-size:12px;color:#666">Grade</div><div class="grade-{grade}">{grade}</div></div>', unsafe_allow_html=True)
                    g2.metric("Tokens originais", data["original_tokens"])
                    g3.metric("Tokens otimizados", data["optimized_tokens"])
                    g4.metric("Tokens economizados", f"-{data['tokens_saved']}", delta=f"-{data['reduction_pct']}%")
                    g5.metric("Custo economizado", f"${data['cost_saved_usd']:.6f}")

                    st.markdown("---")

                    # Prompts lado a lado
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**📝 Prompt original**")
                        st.markdown(f'<div class="prompt-box">{data["original_prompt"]}</div>', unsafe_allow_html=True)

                    with c2:
                        st.markdown("**✅ Prompt otimizado**")
                        st.markdown(f'<div class="optimized-box">{data["optimized_prompt"]}</div>', unsafe_allow_html=True)
                        if st.button("📋 Copiar prompt otimizado"):
                            st.code(data["optimized_prompt"])

                    # Scores
                    st.markdown("---")
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Score de intenção", f"{data['intention_score']:.1%}",
                              help="Similaridade semântica entre prompt original e otimizado")
                    s2.metric("Similaridade de output", f"{data['output_similarity']:.1%}" if test_outputs else "N/A",
                              help="Similaridade entre os outputs do LLM com cada prompt")
                    s3.metric("Técnicas aplicadas", len(data["techniques_applied"]))

                    if data["techniques_applied"]:
                        st.markdown("**🛠️ Técnicas aplicadas:**")
                        for t in data["techniques_applied"]:
                            st.markdown(f"- {t}")

                    # Outputs comparados
                    if test_outputs and data["original_output"]:
                        st.markdown("---")
                        st.markdown("### 🔬 Comparação de outputs")
                        o1, o2 = st.columns(2)
                        with o1:
                            st.markdown("**Output — prompt original:**")
                            st.text_area("", data["original_output"], height=150, key="orig_out")
                        with o2:
                            st.markdown("**Output — prompt otimizado:**")
                            st.text_area("", data["optimized_output"], height=150, key="opt_out")

                else:
                    st.error(f"Erro: {resp.json().get('detail', 'Erro desconhecido')}")

            except requests.exceptions.ConnectionError:
                st.error("API não está rodando. Execute: `cd api && uvicorn main:app --reload`")

    elif optimize_btn:
        st.warning("Cole um prompt para otimizar.")

# ─── TAB 2 — BENCHMARK ───
with tab2:
    st.markdown("### 📊 Benchmark — Dataset de prompts reais")
    st.info("Executa o otimizador nos 6 prompts do dataset de benchmark e exibe métricas comparativas.")

    if st.button("▶️ Executar benchmark", type="primary"):
        with st.spinner("Executando benchmark em 6 prompts..."):
            try:
                resp = requests.get(f"{API_URL}/benchmark", timeout=120)
                if resp.status_code == 200:
                    data = resp.json()

                    st.metric("Redução média de tokens", f"{data['avg_reduction_pct']}%")
                    st.markdown("---")

                    for r in data["results"]:
                        if "error" not in r:
                            grade_color = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🔴"}.get(r["grade"], "⚪")
                            with st.expander(f"{grade_color} {r['id']} — {r['description']} | -{r['reduction_pct']}% tokens"):
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Original", r["original_tokens"])
                                c2.metric("Otimizado", r["optimized_tokens"])
                                c3.metric("Redução", f"{r['reduction_pct']}%")
                                c4.metric("Intenção", f"{r['intention_score']:.1%}")
            except:
                st.error("API não está rodando.")

# ─── TAB 3 — TÉCNICAS ───
with tab3:
    st.markdown("### 📚 Técnicas de otimização implementadas")

    try:
        resp = requests.get(f"{API_URL}/techniques", timeout=5)
        if resp.status_code == 200:
            techniques = resp.json()["techniques"]
            for t in techniques:
                st.markdown(f"**{t['name']}**")
                st.code(t["example"])
                st.markdown("---")
    except:
        techniques = [
            ("Remove filler phrases", "'Please help me' → removed"),
            ("Imperative form", "'Can you summarize' → 'Summarize'"),
            ("Remove AI acknowledgments", "'As an AI language model' → removed"),
            ("Consolidate instructions", "3 similar rules → 1 clear rule"),
            ("Remove obvious context", "'It's important that you' → removed"),
            ("Compress examples", "3 examples → 1 best example"),
            ("Structured format", "prose requirements → bullet list"),
        ]
        for name, example in techniques:
            st.markdown(f"**{name}**")
            st.code(example)
            st.markdown("---")
