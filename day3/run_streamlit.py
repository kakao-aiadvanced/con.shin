import streamlit as st
from app import get_app

def __main__():
    rag_graph_app = get_app()
    # Streamlit 앱 UI
    st.title("Research Assistant powered by OpenAI")

    input_topic = st.text_input(
        ":female-scientist: Enter a topic",
        value="Where does Messi play right now?",
    )

    generate_report = st.button("Generate Report")

    if generate_report:
        with st.spinner("Generating Report"):
            inputs = {"question": input_topic}
            for output in rag_graph_app.stream(inputs):
                for key, value in output.items():
                    print(f"Finished running: {key}:")
            
            if value["hallucination"] == "failed":
                final_report = "failed: hallucination"
            else:
                final_report = value["generation"]
            st.markdown(final_report)

    st.sidebar.markdown("---")
    if st.sidebar.button("Restart"):
        st.session_state.clear()
        st.experimental_rerun()

__main__()