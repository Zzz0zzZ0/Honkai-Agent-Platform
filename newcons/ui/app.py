import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Neuromorphic Brain", page_icon="🧠", layout="wide")
st.title("🧠 智能运营大盘：RAG-Agent x 情感画像")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "viz_data" not in st.session_state:
    st.session_state.viz_data = None

with st.sidebar:
    st.header("🎛️ 运营中枢控制台")
    use_agent = st.toggle("🌐 启用 Agent 工作流 (自主路由/公关/看板)", value=True)
    st.divider()
    selected_model = (
        "cloud"
        if "云端" in st.radio("推理模型", ("☁️ 云端 Gemini", "💻 本地 Qwen"))
        else "local"
    )
    use_auto_alpha = st.toggle("🤖 LinUCB 自适应学习", value=True)
    alpha = 0.5 if use_auto_alpha else st.slider("Alpha 权重", 0.0, 1.0, 0.5)

    k_val = st.slider("工作记忆 (Top-K)", 1, 6, 3)
    temp_val = st.slider("发散度 (Temperature)", 0.0, 1.0, 0.1)

    st.divider()
    use_emotion = st.toggle("🧠 开启结构化感知 (画像打标&舆情落表)", value=True)

    st.subheader("📊 舆情大盘看板")
    if st.button("生成社区报告"):
        if os.path.exists("community_feedback_log.csv"):
            df_log = pd.read_csv("community_feedback_log.csv")
            st.warning(f"⚠️ 拦截到 {len(df_log)} 条高危客诉！")
            st.dataframe(df_log, use_container_width=True)
        else:
            st.success("✅ 社区情绪稳定。")

    uploaded_file = st.file_uploader("📂 记忆注入 (文档)", type=["pdf", "txt"])
    if uploaded_file:
        if st.button("🚀 上传并解析"):
            with st.spinner("正通过 API 发送至后端引擎..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "multipart/form-data",
                    )
                }
                res = requests.post(f"{API_URL}/upload_memory", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.success(data["message"])
                    if data.get("viz_data"):
                        st.session_state.viz_data = pd.DataFrame(data["viz_data"])
                else:
                    st.error(f"上传失败: {res.text}")

col_chat, col_viz = st.columns([1, 1])

with col_viz:
    st.subheader("🌌 语义空间可视化")
    if st.session_state.viz_data is not None:
        fig = px.scatter(
            st.session_state.viz_data,
            x="x",
            y="y",
            hover_data=["text"],
            color_discrete_sequence=["#FF4B4B"],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("上传文档后将在此生成记忆神经元的 2D 降维映射。")

with col_chat:
    st.subheader("💬 认知交互界面")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("输入指令或问题..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("⚡ 后端 API 推理中..."):
                payload = {
                    "query": prompt,
                    "use_agent": use_agent,
                    "model_type": selected_model,
                    "use_auto_alpha": use_auto_alpha,
                    "alpha": alpha,
                    "use_emotion": use_emotion,
                    "k_param": k_val,
                    "temp_param": temp_val,
                }
                try:
                    response = requests.post(f"{API_URL}/chat", json=payload)
                    if response.status_code == 200:
                        data = response.json()

                        if data.get("thoughts"):
                            with st.status("🧠 Agent 思考流...", expanded=False):
                                for t in data["thoughts"]:
                                    st.write(t)

                        if data.get("persona"):
                            persona_html = " ".join(
                                [
                                    "<span style='background:#ffe0b2;color:#e65100;"
                                    "padding:2px 8px;border-radius:12px;"
                                    "font-weight:bold;font-size:0.85em'>🎯 "
                                    f"{p}</span>"
                                    for p in data["persona"]
                                ]
                            )
                            st.markdown(
                                f"**👤 捕获玩家画像**: {persona_html}",
                                unsafe_allow_html=True,
                            )

                        st.markdown(data["answer"])
                        st.session_state.messages.append(
                            {"role": "assistant", "content": data["answer"]}
                        )
                    else:
                        st.error(f"后端报错: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("🚨 无法连接到 FastAPI 后端！请确保后端服务正在 8000 端口运行。")

