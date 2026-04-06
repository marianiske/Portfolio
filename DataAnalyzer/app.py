import streamlit as st
from agent.LLM import LLM

st.set_page_config(page_title="Data Analyst", page_icon="📊", layout="wide")
st.title("📊 Data Analyst")

if "llm" not in st.session_state:
    st.session_state.llm = LLM()
if "messages" not in st.session_state:
    st.session_state.messages = []

llm: LLM = st.session_state.llm

with st.sidebar:
    st.header("Current dataset")

    if llm.current_dataset_name:
        st.success(f"**{llm.current_dataset_name}**")
        df = llm.loaded_datasets[llm.current_dataset_name]
        col1, col2 = st.columns(2)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])

        with st.expander("Preview (first 20 rows)"):
            st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("No dataset loaded yet.\nAsk me to load one!")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        llm.reset_conversation()
        st.rerun()
        
    st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)
    st.caption("Author: Marian Iske")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("image_path"):
            try:
                st.image(msg["image_path"])
            except Exception:
                pass

if prompt := st.chat_input("Ask me anything about your data…"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = llm.run(prompt)

        st.write(answer)

        trace = llm.last_run_trace

        image_path = None
        scatter_results = [
            r for r in trace.get("tool_results", [])
            if r["name"] == "scatter_plot"
        ]
        if scatter_results:
            path = scatter_results[-1]["result"].get("plot_path")
            if path:
                try:
                    st.image(path)
                    image_path = path
                except Exception:
                    pass

        if trace.get("tool_calls"):
            with st.expander("🔧 Tool calls", expanded=False):
                for call in trace["tool_calls"]:
                    args = call.get("arguments") or {}
                    label = f"`{call['name']}`"
                    if args:
                        label += "  —  " + ", ".join(
                            f"**{k}** = `{v}`" for k, v in args.items()
                        )
                    st.markdown(label)

        if trace.get("errors"):
            with st.expander("⚠️ Errors", expanded=True):
                for err in trace["errors"]:
                    st.error(err)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "image_path": image_path,
    })
