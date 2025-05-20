
# import os
# # Must be set BEFORE importing streamlit
# os.environ["STREAMLIT_WATCHER_TYPE"] = "watchdog"
# os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# import streamlit as st
# from agents import Main_agent, load_faiss_index
# import time

# # Set page config
# st.set_page_config(
#     page_title="Healthcare Research Assistant",
#     page_icon="🏥",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize session state
# if 'faiss_index' not in st.session_state:
#     with st.spinner("Loading research database..."):
#         st.session_state.faiss_index = load_faiss_index()
# if 'history' not in st.session_state:
#     st.session_state.history = []

# # Sidebar
# with st.sidebar:
#     st.image("https://cdn-icons-png.flaticon.com/512/2961/2961287.png", width=80)
#     st.title("Healthcare Research Assistant")
#     st.markdown("""
#     This AI assistant can help with:
#     - Medical questions (symptoms, treatments)
#     - Doctor appointment scheduling
#     - General health information
#     """)
    
#     st.markdown("---")
#     st.markdown("### Conversation History")
#     for i, item in enumerate(st.session_state.history):
#         if st.sidebar.button(f"Q: {item['query'][:30]}...", key=f"hist_{i}"):
#             st.session_state.current_query = item['query']
#             st.session_state.current_response = item['response']

# # Main content
# st.title("Healthcare Research Assistant")
# st.markdown("Ask about medical conditions, doctor availability, or general health questions.")

# query = st.text_input(
#     "Enter your healthcare-related question:",
#     value=getattr(st.session_state, 'current_query', ''),
#     key="input_query",
#     placeholder="E.g. What are the symptoms of diabetes? or Is Dr. Smith available on Friday?"
# )

# if st.button("Get Answer"):
#     if query.strip() == "":
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Researching your question..."):
#             start_time = time.time()

#             progress_bar = st.progress(0)
#             status_text = st.empty()

#             for i in range(5):
#                 progress_bar.progress((i + 1) * 20)
#                 status_text.text(f"Processing... Step {i+1}/5")
#                 time.sleep(0.2)

#             response = Main_agent(query, st.session_state.faiss_index)

#             st.session_state.history.append({
#                 "query": query,
#                 "response": response
#             })
#             st.session_state.current_query = query
#             st.session_state.current_response = response

#             status_text.empty()
#             progress_bar.empty()

#             # st.subheader("Response:")
#             # st.markdown(response)
#             st.caption(f"Response generated in {time.time() - start_time:.2f} seconds")

# if hasattr(st.session_state, 'current_response'):
#     st.subheader("Response:")
#     st.markdown(st.session_state.current_response)



import os
# Must be set BEFORE importing streamlit
os.environ["STREAMLIT_WATCHER_TYPE"] = "watchdog"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
from agents import Main_agent, load_faiss_index
from data_scout import DataScout_agent, DataScout_agent_with_pdf
import time
import pandas as pd
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Healthcare Research Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'faiss_index' not in st.session_state:
    with st.spinner("Loading research database..."):
        st.session_state.faiss_index = load_faiss_index()
if 'history' not in st.session_state:
    st.session_state.history = []
if 'data_creation_type' not in st.session_state:
    st.session_state.data_creation_type = "Excel"
if 'generated_file' not in st.session_state:
    st.session_state.generated_file = None

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2961/2961287.png", width=80)
    st.title("Healthcare Research Assistant")
    st.markdown("""
    This AI assistant can help with:
    - Medical questions (symptoms, treatments)
    - Doctor appointment scheduling
    - General health information
    - Data creation (Excel/PDF reports)
    """)
    
    st.markdown("---")
    st.markdown("### Conversation History")
    for i, item in enumerate(st.session_state.history):
        if st.sidebar.button(f"Q: {item['query'][:30]}...", key=f"hist_{i}"):
            st.session_state.current_query = item['query']
            st.session_state.current_response = item['response']

# Main content with tabs
tab1, tab2 = st.tabs(["Healthcare Assistant", "Data Creation"])

with tab1:
    st.title("Healthcare Research Assistant")
    st.markdown("Ask about medical conditions, doctor availability, or general health questions.")

    query = st.text_input(
        "Enter your healthcare-related question:",
        value=getattr(st.session_state, 'current_query', ''),
        key="input_query",
        placeholder="E.g. What are the symptoms of diabetes? or Is Dr. Smith available on Friday?"
    )

    if st.button("Get Answer"):
        if query.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Researching your question..."):
                start_time = time.time()

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(5):
                    progress_bar.progress((i + 1) * 20)
                    status_text.text(f"Processing... Step {i+1}/5")
                    time.sleep(0.2)

                response = Main_agent(query, st.session_state.faiss_index)

                st.session_state.history.append({
                    "query": query,
                    "response": response
                })
                st.session_state.current_query = query
                st.session_state.current_response = response

                status_text.empty()
                progress_bar.empty()

                st.caption(f"Response generated in {time.time() - start_time:.2f} seconds")

    if hasattr(st.session_state, 'current_response'):
        st.subheader("Response:")
        st.markdown(st.session_state.current_response)

with tab2:
    st.title("Data Creation Assistant")
    st.markdown("Create Excel spreadsheets or PDF reports with AI")

    # Data type selection
    data_type = st.radio(
        "Select data type to create:",
        ("Excel", "PDF"),
        horizontal=True,
        key="data_creation_type"
    )
    
    # Input fields based on selection
    if st.session_state.data_creation_type == "Excel":
        prompt = st.text_area(
            "Describe the data you want to generate (include field names and number of rows):",
            placeholder="E.g. Generate 25 patient records with fields: Name, Age, Gender, Diagnosis, Admission Date"
        )
    else:
        prompt = st.text_area(
            "Describe the PDF document you want to generate (include sections and content requirements):",
            placeholder="E.g. Create a 3-page medical report about diabetes with sections: Introduction, Symptoms, Treatment Options"
        )

    if st.button("Generate Data"):
        if prompt.strip() == "":
            st.warning("Please enter a description of what you want to generate")
        else:
            with st.spinner(f"Generating {st.session_state.data_creation_type} file..."):
                try:
                    if st.session_state.data_creation_type == "Excel":
                        agent = DataScout_agent()
                        response = agent.invoke(prompt)
                        if isinstance(response, dict) and 'output' in response:
                            file_path = response['output']
                            st.session_state.generated_file = {
                                'type': 'excel',
                                'path': file_path,
                                'name': os.path.basename(file_path)
                            }
                        else:
                            st.error("Failed to generate Excel file")
                    else:
                        agent = DataScout_agent_with_pdf()
                        response = agent.invoke(prompt)
                        if isinstance(response, dict) and 'output' in response:
                            file_path = response['output']
                            st.session_state.generated_file = {
                                'type': 'pdf',
                                'path': file_path,
                                'name': os.path.basename(file_path)
                            }
                        else:
                            st.error("Failed to generate PDF file")
                except Exception as e:
                    st.error(f"Error generating file: {str(e)}")

    # Download button for generated file
    if st.session_state.generated_file:
        st.success(f"{st.session_state.data_creation_type} file generated successfully!")
        
        if st.session_state.generated_file['type'] == 'excel':
            df = pd.read_excel(st.session_state.generated_file['path'])
            st.dataframe(df.head())
            
            with open(st.session_state.generated_file['path'], "rb") as f:
                bytes_data = f.read()
            
            st.download_button(
                label="Download Excel File",
                data=bytes_data,
                file_name=st.session_state.generated_file['name'],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            with open(st.session_state.generated_file['path'], "rb") as f:
                bytes_data = f.read()
            
            st.download_button(
                label="Download PDF File",
                data=bytes_data,
                file_name=st.session_state.generated_file['name'],
                mime="application/pdf"
            )

        if st.button("Clear Generated File"):
            st.session_state.generated_file = None
            st.rerun()