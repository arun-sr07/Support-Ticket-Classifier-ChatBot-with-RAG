import streamlit as st
import pandas as pd
import time
import base64
import os
from vectors import EmbeddingsManager
from chatbot import ChatbotManager
from helpdesk_analytics import HelpdeskAnalytics
from network import NetworkAnalytics

# Function to display PDFs
def displayPDF(file):
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to display CSV files
def displayCSV(file):
    # Try different encodings if UTF-8 fails
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            # Reset file pointer to beginning
            file.seek(0)
            df = pd.read_csv(file, encoding=encoding)
            st.dataframe(df)  # Show CSV content in table format
            return  # Successfully read the file, exit the function
        except UnicodeDecodeError:
            continue  # Try the next encoding
        except Exception as e:
            st.error(f"Error reading CSV with {encoding} encoding: {e}")
            continue
    
    # If all encodings fail, show a generic error
    st.error("Could not read the CSV file. Please check if the file is properly formatted.")

# Initialize session state
if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'analytics' not in st.session_state:
    st.session_state['analytics'] = None
if 'network_analytics' not in st.session_state:
    st.session_state['network_analytics'] = None

# Sidebar

st.sidebar.markdown("### Support Ticket Classifier ChatBot")
menu = ["🏠 Home", "🤖 Chatbot", "📊 Helpdesk Analytics"]
choice = st.sidebar.selectbox("Navigate", menu)

# Home Page
if choice == "🏠 Home":
    st.title("📄 Support Ticket Classifier ChatBot")
    st.markdown("Supports both **PDF and CSV** files! 🚀")

# Chatbot Page
elif choice == "🤖 Chatbot":
    st.title("🤖 Chatbot Interface")
    #col1, col2 = st.columns(2)


    col1, col2, col3 = st.columns(3)

    # Column 1: File Uploader
    with col1:
        st.header("📂 Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF or CSV", type=["pdf", "csv"])

        if uploaded_file is not None:
            st.success(f"📄 File '{uploaded_file.name}' Uploaded Successfully!")

            file_type = uploaded_file.type
            st.session_state['file_type'] = file_type

            if file_type == "application/pdf":
                st.markdown("### 📖 PDF Preview")
                displayPDF(uploaded_file)
                temp_path = "temp.pdf"
            elif file_type == "text/csv":
                st.markdown("### 📊 CSV Preview")
                displayCSV(uploaded_file)
                temp_path = "temp.csv"

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state['temp_file_path'] = temp_path

    # Column 2: Create Embeddings
    with col2:
        st.header(" Create Embeddings")
        create_embeddings = st.checkbox("✅ Generate Embeddings")

        if create_embeddings:
            if st.session_state['temp_file_path'] is None:
                st.warning("⚠️ Please upload a document first.")
            else:
                try:
                    embeddings_manager = EmbeddingsManager()

                    with st.spinner("🔄 Creating Embeddings..."):
                        result = embeddings_manager.create_embeddings(st.session_state['temp_file_path'])
                        time.sleep(1)
                    st.success(result)

                    # Initialize ChatbotManager
                    if st.session_state['chatbot_manager'] is None:
                        st.session_state['chatbot_manager'] = ChatbotManager()
                        
                    # Initialize HelpdeskAnalytics if it's a CSV file
                    if st.session_state['file_type'] == "text/csv" and st.session_state['analytics'] is None:
                        st.session_state['analytics'] = HelpdeskAnalytics(st.session_state['temp_file_path'])
                        
                    # Initialize NetworkAnalytics if it's a CSV file
                    if st.session_state['file_type'] == "text/csv" and st.session_state['network_analytics'] is None:
                        st.session_state['network_analytics'] = NetworkAnalytics(st.session_state['temp_file_path'])

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # Column 3: Chatbot Interface
    with col3:
        st.header("💬 Chat with Document")

        if st.session_state['chatbot_manager'] is None:
            st.info("🤖 Upload a file and generate embeddings to chat.")
        else:
            for msg in st.session_state['messages']:
                st.chat_message(msg['role']).markdown(msg['content'])

            if user_input := st.chat_input("Type your message..."):
                st.chat_message("user").markdown(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})

                with st.spinner("🤖 Responding..."):
                    try:
                        answer = st.session_state['chatbot_manager'].get_response(user_input)
                        time.sleep(1)
                    except Exception as e:
                        answer = f"⚠️ Error: {e}"

                st.chat_message("assistant").markdown(answer)
                st.session_state['messages'].append({"role": "assistant", "content": answer})

# Helpdesk Analytics Page
elif choice == "📊 Helpdesk Analytics":
    st.title("📊 Helpdesk Analytics")
    
    if st.session_state['analytics'] is None:
        st.info("⚠️ Please upload a CSV file and generate embeddings in the Chatbot section first.")
    else:
        st.success("✅ Analytics ready! Ask questions about your helpdesk data.")
        
        # Suggested visualization queries
        st.subheader("Suggested Visualization Queries:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Priority Analysis:**")
            if st.button("Show tickets by priority (Bar Chart)"):
                response = st.session_state['analytics'].handle_query("Show me a bar chart of tickets by priority")
                st.markdown(response, unsafe_allow_html=True)
                
            if st.button("Show ticket distribution by priority (Pie Chart)"):
                response = st.session_state['analytics'].handle_query("Create a pie chart of ticket distribution by priority")
                st.markdown(response, unsafe_allow_html=True)
                
            st.markdown("**Satisfaction Analysis:**")
            if st.button("Show customer satisfaction levels (Bar Chart)"):
                response = st.session_state['analytics'].handle_query("Create a bar chart of customer satisfaction levels")
                st.markdown(response, unsafe_allow_html=True)
                
            if st.button("Show satisfaction distribution (Pie Chart)"):
                response = st.session_state['analytics'].handle_query("Create a pie chart of customer satisfaction distribution")
                st.markdown(response, unsafe_allow_html=True)
                
        with col2:
            st.markdown("**Resolution Time Analysis:**")
            if st.button("Show ticket resolution time distribution (Histogram)"):
                response = st.session_state['analytics'].handle_query("Show me a histogram of ticket resolution time")
                st.markdown(response, unsafe_allow_html=True)
                
            if st.button("Show resolution time by priority (Box Plot)"):
                response = st.session_state['analytics'].handle_query("Show me a box plot of days open by priority")
                st.markdown(response, unsafe_allow_html=True)
                
            st.markdown("**Category Analysis:**")
            if st.button("Show tickets by category (Bar Chart)"):
                response = st.session_state['analytics'].handle_query("Display a bar chart of tickets by category")
                st.markdown(response, unsafe_allow_html=True)
                
            if st.button("Show ticket type distribution (Pie Chart)"):
                response = st.session_state['analytics'].handle_query("Show me a pie chart of ticket types")
                st.markdown(response, unsafe_allow_html=True)
        
        # Custom query input
        st.subheader("Custom Visualization Query:")
        custom_query = st.text_input("Enter your visualization query", 
                                    placeholder="e.g., Show me a bar chart of tickets by severity")
        
        if st.button("Generate Visualization"):
            if custom_query:
                with st.spinner("🔄 Generating visualization..."):
                    response = st.session_state['analytics'].handle_query(custom_query)
                    st.markdown(response, unsafe_allow_html=True)
            else:
                st.warning("Please enter a query first.")


