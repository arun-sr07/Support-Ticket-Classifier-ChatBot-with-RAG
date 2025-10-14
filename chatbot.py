from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from helpdesk_analytics import HelpdeskAnalytics
import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
load_dotenv()

class ChatbotManager:
    def __init__(self, csv_path=None):
        self.analytics = HelpdeskAnalytics(csv_path) if csv_path else None
        

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"device": "cpu"}
        )

        # ✅ Replace Ollama with Groq LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # ✅ Qdrant setup
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "vector_db"
        self.db = Qdrant(client=self.client, embeddings=self.embeddings, collection_name=self.collection_name)

        self.prompt = PromptTemplate(
            template="You are an AI assistant answering questions based on the latest uploaded document only and if anything apart from the document the question asked respond I am not aware of it.\n"
                     "Document Context:\n{context}\n\n"
                     "User Question:\n{question}\n\n"
                     "Helpful and precise answer:",
            input_variables=["context", "question"]
        )

        keyword_filters = Filter(should=[
            FieldCondition(key="FiledAgainst", match=MatchValue(value="Access/Login")),
            FieldCondition(key="Priority", match=MatchValue(value="2 - High"))
        ])

        self.retriever = self.db.as_retriever(search_kwargs={
            "k": 3,
            "filter": keyword_filters
        })

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def get_response(self, query: str) -> str:
        query_lower = query.lower()
        
        # Check for helpdesk-related keywords first
        helpdesk_keywords = ["ticket", "issue", "priority", "satisfaction", "resolution", "days open", "helpdesk", "total", "how many", "count", "high priority", "critical", "medium", "low"]
        
        # If it's a helpdesk query and we have analytics, use it
        if self.analytics and any(keyword in query_lower for keyword in helpdesk_keywords):
            return self.analytics.handle_query(query)
        
        # If it's a helpdesk query but no analytics, provide basic analysis
        if any(keyword in query_lower for keyword in helpdesk_keywords):
            return self._handle_helpdesk_query_without_analytics(query)
        
        # For non-helpdesk queries, use the document QA system
        try:
            response = self.qa.run(query)
            return response if response.strip() else "I couldn't find relevant information in the uploaded document."
        except Exception as e:
            return f"⚠️ Error: {e}"
    
    def _handle_helpdesk_query_without_analytics(self, query: str) -> str:
        """Handle helpdesk queries when analytics is not available"""
        try:
            import pandas as pd
            df = pd.read_csv('temp.csv')
            
            query_lower = query.lower()
            
            if "total" in query_lower and "ticket" in query_lower:
                total_tickets = len(df)
                return f"Total number of helpdesk tickets: {total_tickets}"
            
            elif "high priority" in query_lower or "critical" in query_lower:
                if "high priority" in query_lower:
                    high_priority = len(df[df['Ticket Priority'] == 'High'])
                    return f"Total tickets marked as High priority: {high_priority}"
                else:
                    critical_priority = len(df[df['Ticket Priority'] == 'Critical'])
                    return f"Total tickets marked as Critical priority: {critical_priority}"
            
            elif "priority" in query_lower:
                priority_counts = df['Ticket Priority'].value_counts()
                result = "Ticket Priority Distribution:\n"
                for priority, count in priority_counts.items():
                    result += f"- {priority}: {count} tickets\n"
                return result
            
            elif "status" in query_lower:
                status_counts = df['Ticket Status'].value_counts()
                result = "Ticket Status Distribution:\n"
                for status, count in status_counts.items():
                    result += f"- {status}: {count} tickets\n"
                return result
            
            elif "type" in query_lower:
                type_counts = df['Ticket Type'].value_counts()
                result = "Ticket Type Distribution:\n"
                for ticket_type, count in type_counts.items():
                    result += f"- {ticket_type}: {count} tickets\n"
                return result
            
            else:
                return "I can help you analyze helpdesk data. Try asking about ticket counts, priorities, status, or types."
                
        except Exception as e:
            return f"⚠️ Error analyzing helpdesk data: {e}"
