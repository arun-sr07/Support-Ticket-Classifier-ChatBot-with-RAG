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
import json

load_dotenv()

class ChatbotManager:
    def __init__(self, csv_path=None):
        self.analytics = HelpdeskAnalytics(csv_path) if csv_path else None

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"device": "cpu"}
        )

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "vector_db"
        self.db = Qdrant(client=self.client, embeddings=self.embeddings, collection_name=self.collection_name)

        self.prompt = PromptTemplate(
            template="You are a helpdesk AI assistant with access to both analytics data and document knowledge.\n"
        "Use the analytics summary for quantitative insights (like counts, types, and patterns)\n"
        "and the document context for procedural or policy-based explanations.\n\n"
        "If analytics_summary is empty, rely solely on the document context.\n"
        "If both are available, combine them logically to give a complete and practical answer.\n\n"
        "=== Analytics Summary ===\n"
        "{analytics_summary}\n\n"
        "=== Document Context ===\n"
        "{context}\n\n"
        "=== User Question ===\n"
        "{question}\n\n"
        "Now write a clear, step-by-step, and well-grounded answer that combines both insights.\n"
        "Be concise, accurate, and professional.",
            input_variables=["context", "question","analytics_summary"]
        )
        self.offensive_words = ["idiot", "stupid", "dumb", "fool", "nonsense", "useless", "shut up"]

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
    def _contains_offensive_language(self, text):
        text_lower = text.lower()
        return any(word in text_lower for word in self.offensive_words)
    def get_response(self, query: str) -> str:
        query_lower = query.lower()
        if self._contains_offensive_language(query_lower):
            return (
                "Let's keep it professional. Please rephrase your question politely so I can assist you effectively."
            )
        # Helpdesk-related keywords
        helpdesk_keywords = ["ticket", "issue", "priority", "satisfaction", "resolution",
                             "days open", "helpdesk", "total", "how many", "count",
                             "high priority", "critical", "medium", "low"]
        
        # --- Hybrid Helpdesk + Document reasoning mode ---
        combined_keywords = ["how", "solved", "resolved", "handled", "why", "fix", "solution", "steps"]
        if any(k in query_lower for k in helpdesk_keywords) and any(k in query_lower for k in combined_keywords):
            try:
                # Step 1: Get analytics summary
                analytics_summary = ""
                if self.analytics:
                    analytics_summary = self.analytics.handle_query("most common ticket issues")
                else:
                    analytics_summary = self._handle_helpdesk_query_without_analytics("most common ticket issues")

                # Step 2: Retrieve related document context
                docs = self.retriever.get_relevant_documents(query)
                doc_context = "\n\n".join([d.page_content[:600] for d in docs[:3]])

                # Step 3: Combine both sources for contextual answer
                hybrid_prompt = (
                    "You are a helpdesk expert. Combine the analytical summary below with document knowledge "
                    "to explain what the most common issues are and how they are solved.\n\n"
                    f"Analytical Summary:\n{analytics_summary}\n\n"
                    f"Document Context:\n{doc_context}\n\n"
                    "Write a short, clear explanation (4–7 sentences) that includes both patterns and solution steps."
                )

                result = self.llm.invoke(hybrid_prompt)
                if hasattr(result, "content"):
                    return result.content
                return result
            except Exception as e:
                return f"⚠️ Hybrid reasoning failed: {e}"

        # --- Analytics-only mode ---
        if self.analytics and any(keyword in query_lower for keyword in helpdesk_keywords):
            return self.analytics.handle_query(query)

        # --- Fallback analytics if CSV not connected ---
        if any(keyword in query_lower for keyword in helpdesk_keywords):
            return self._handle_helpdesk_query_without_analytics(query)

        # --- Default RAG flow (document-based QA + classification) ---
        try:
            result = self.categorize_and_draft(query)
            if isinstance(result, dict) and not result.get("error"):
                parts = []
                parts.append(
                    f"**Filed Against:** {result.get('filedAgainst','N/A')}  |  "
                    f"**Type:** {result.get('ticketType','N/A')}  |  "
                    f"**Severity:** {result.get('severity','N/A')}  |  "
                    f"**Priority:** {result.get('priority','N/A')}"
                )
                parts.append("")
                parts.append(result.get("reply", ""))
                cites = result.get("citations", [])
                if cites:
                    parts.append("\nReferences:")
                    for i, c in enumerate(cites[:3], start=1):
                        snippet = c.get("snippet", "")
                        parts.append(f"{i}. {snippet}")
                text = "\n".join(parts).strip()
                return text if text else "I couldn't generate a grounded reply from the current context."
            
            # Fallback QA
            response = self.qa.run(query)
            return response if response.strip() else "I couldn't find relevant information in the uploaded document."
        except Exception as e:
            return f"⚠️ Error: {e}"

    def categorize_and_draft(self, message: str) -> dict:
        try:
            docs = self.retriever.get_relevant_documents(message)
            top_docs = docs[:3]
            context_blocks = []
            for d in top_docs:
                text = (d.page_content or "").strip()
                if len(text) > 600:
                    text = text[:600] + "…"
                context_blocks.append(text)
            context = "\n\n".join(context_blocks) if context_blocks else ""

            system_instructions = (
                "You are a support assistant. Classify the user's issue and draft a concise reply. "
                "Use the provided context only for facts. If context is missing, make a best-effort classification but avoid hallucinations. "
                "Output STRICT JSON with keys: filedAgainst, ticketType, severity, priority, reply. "
                "Allowed values (best effort):\n"
                "- filedAgainst: [Access/Login, Systems, Software, Hardware, Other]\n"
                "- ticketType: [Request, Issue, Cancellation request, Refund request, Product inquiry, Other]\n"
                "- severity: [1 - Minor, 2 - Normal, 3 - Major, 4 - Critical]\n"
                "- priority: [0 - Unassigned, 1 - Low, 2 - Medium, 3 - High]\n"
                "reply should be grounded, 3-6 sentences, with specific steps."
            )

            prompt = (
                f"{system_instructions}\n\n"
                f"Context:\n{context}\n\n"
                f"User Issue:\n{message}\n\n"
                "Return JSON only."
            )

            raw = self.llm.invoke(prompt)
            output_text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))

            result = {}
            try:
                result = json.loads(output_text)
            except Exception:
                start = output_text.find("{")
                end = output_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    result = json.loads(output_text[start:end+1]) if output_text[start:end+1] else {}

            filed_against = result.get("filedAgainst", "Other")
            ticket_type = result.get("ticketType", "Issue")
            severity = result.get("severity", "2 - Normal")
            priority = result.get("priority", "2 - Medium")
            reply = result.get("reply", "I couldn't generate a grounded reply from the current context.")

            citations = []
            for d in top_docs:
                snippet = (d.page_content or "").strip()
                if len(snippet) > 220:
                    snippet = snippet[:220] + "…"
                citations.append({
                    "snippet": snippet,
                    "metadata": d.metadata if hasattr(d, "metadata") else {}
                })

            return {
                "filedAgainst": filed_against,
                "ticketType": ticket_type,
                "severity": severity,
                "priority": priority,
                "reply": reply,
                "citations": citations,
            }
        except Exception as e:
            return {"error": str(e)}

    def _handle_helpdesk_query_without_analytics(self, query: str) -> str:
        try:
            import pandas as pd
            df = pd.read_csv('temp.csv')

            query_lower = query.lower()
            if "total" in query_lower and "ticket" in query_lower:
                return f"Total number of helpdesk tickets: {len(df)}"

            elif "high priority" in query_lower or "critical" in query_lower:
                if "high priority" in query_lower:
                    return f"Total tickets marked as High priority: {len(df[df['Ticket Priority'] == 'High'])}"
                else:
                    return f"Total tickets marked as Critical priority: {len(df[df['Ticket Priority'] == 'Critical'])}"

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
