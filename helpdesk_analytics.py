import pandas as pd
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_community.llms import HuggingFaceHub
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.figure import Figure

class HelpdeskAnalytics:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        
        self.llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",  
            huggingfacehub_api_token="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",
            model_kwargs={"temperature": 0.1}
        )
        
        
        def dummy_function(query):
            return "This is a dummy function to avoid API calls. Please use the visualization features instead."
        
        dummy_tool = Tool(
            name="dummy_tool",
            func=dummy_function,
            description="A dummy tool to avoid API calls"
        )
        
        
        self.agent = initialize_agent(
            tools=[dummy_tool], 
            llm=self.llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            verbose=False  
        )
        
        self.helpdesk_keywords = ["ticket", "priority", "unresolved", "days open","satisfaction","ticket type","customer","resolution","status"]

        
        self.offensive_words = ["idiot", "stupid", "dumb", "fool"]
        
        
        plt.style.use('ggplot')
        
    def handle_query(self, query):
        """Processes user queries and lets the AI agent handle mathematical computations autonomously."""
        
        if any(keyword in query.lower() for keyword in ["graph", "chart", "plot", "visual", "bar", "pie", "histogram", "percentage", "what percentage"]):
            return self.create_visualization(query)
        
        try:
           
            response = self.llm.invoke(query)
            return response

        except Exception as e:
            return f"⚠️ Error: {e}"
    
    def create_visualization(self, query):
        """Creates visualizations based on the query and returns the image as a base64 string."""
        query = query.lower()
        
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        
        if "priority" in query:
            if "bar" in query or "chart" in query:
               
                priority_counts = self.df['Ticket Priority'].value_counts()
                priority_counts.plot(kind='bar', ax=ax)
                ax.set_title('Tickets by Priority')
                ax.set_xlabel('Priority')
                ax.set_ylabel('Number of Tickets')
                plt.tight_layout()
            elif "pie" in query:
                # Create a pie chart of ticket distribution by priority
                priority_counts = self.df['Ticket Priority'].value_counts()
                ax.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%')
                ax.set_title('Ticket Distribution by Priority')
        
        elif "satisfaction" in query:
            if "bar" in query or "chart" in query:
                # Create a bar chart of satisfaction levels
                satisfaction_counts = self.df['Customer Satisfaction Rating'].value_counts()
                satisfaction_counts.plot(kind='bar', ax=ax)
                ax.set_title('Customer Satisfaction Levels')
                ax.set_xlabel('Satisfaction Level')
                ax.set_ylabel('Number of Tickets')
                plt.tight_layout()
            elif "pie" in query:
                # Create a pie chart of satisfaction distribution
                satisfaction_counts = self.df['Customer Satisfaction Rating'].value_counts()
                ax.pie(satisfaction_counts, labels=satisfaction_counts.index, autopct='%1.1f%%')
                ax.set_title('Customer Satisfaction Distribution')
            elif "percentage" in query or "what percentage" in query:
                # Calculate percentage of high satisfaction tickets
                satisfaction_counts = self.df['Customer Satisfaction Rating'].value_counts()
                total_tickets = len(self.df)
                
                # Find high satisfaction categories (assuming they contain "satisfied" or "highly satisfied")
                high_satisfaction = 0
                for category, count in satisfaction_counts.items():
                    if "satisfied" in category.lower() or "highly satisfied" in category.lower():
                        high_satisfaction += count
                
                percentage = (high_satisfaction / total_tickets) * 100
                
                # Create a pie chart showing high satisfaction vs others
                labels = ['High Satisfaction', 'Other Satisfaction Levels']
                sizes = [high_satisfaction, total_tickets - high_satisfaction]
                ax.pie(sizes, labels=labels, autopct='%1.1f%%')
                ax.set_title(f'Percentage of Tickets with High Satisfaction: {percentage:.1f}%')
        
        elif "days open" in query or "resolution time" in query:
            if "histogram" in query or "distribution" in query:
                # Create a histogram of days open
                ax.hist(self.df['Time to Resolution'], bins=20, alpha=0.7)
                ax.set_title('Distribution of Ticket Resolution Time')
                ax.set_xlabel('Days Open')
                ax.set_ylabel('Number of Tickets')
                plt.tight_layout()
            elif "box" in query:
                # Create a box plot of days open by priority
                sns.boxplot(x='Ticket Priority', y='Time to Resolution', data=self.df, ax=ax)
                ax.set_title('Resolution Time by Priority')
                ax.set_xlabel('Priority')
                ax.set_ylabel('Days Open')
                plt.tight_layout()
        
        elif "filed against" in query or "category" in query:
            if "bar" in query or "chart" in query:
                # Create a bar chart of tickets by category
                category_counts = self.df['Ticket Type'].value_counts()
                category_counts.plot(kind='bar', ax=ax)
                ax.set_title('Tickets by Category')
                ax.set_xlabel('Category')
                ax.set_ylabel('Number of Tickets')
                plt.tight_layout()
            elif "pie" in query:
                # Create a pie chart of ticket distribution by category
                category_counts = self.df['Ticket Type'].value_counts()
                ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
                ax.set_title('Ticket Distribution by Category')
        
        elif "ticket type" in query:
            if "bar" in query or "chart" in query:
                # Create a bar chart of tickets by type
                type_counts = self.df['TicketType'].value_counts()
                type_counts.plot(kind='bar', ax=ax)
                ax.set_title('Tickets by Type')
                ax.set_xlabel('Ticket Type')
                ax.set_ylabel('Number of Tickets')
                plt.tight_layout()
            elif "pie" in query:
                # Create a pie chart of ticket distribution by type
                type_counts = self.df['TicketType'].value_counts()
                ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
                ax.set_title('Ticket Distribution by Type')
        
        elif "severity" in query:
            if "bar" in query or "chart" in query:
                # Create a bar chart of tickets by severity
                severity_counts = self.df['Ticket Priority'].value_counts()
                severity_counts.plot(kind='bar', ax=ax)
                ax.set_title('Tickets by Severity')
                ax.set_xlabel('Severity')
                ax.set_ylabel('Number of Tickets')
                plt.tight_layout()
            elif "pie" in query:
                # Create a pie chart of ticket distribution by severity
                severity_counts = self.df['Ticket Priority'].value_counts()
                ax.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%')
                ax.set_title('Ticket Distribution by Severity')
        
        elif "requestor seniority" in query:
            if "bar" in query or "chart" in query:
                # Create a bar chart of tickets by requestor seniority
                seniority_counts = self.df['Customer Age'].value_counts()
                seniority_counts.plot(kind='bar', ax=ax)
                ax.set_title('Tickets by Requestor Seniority')
                ax.set_xlabel('Seniority Level')
                ax.set_ylabel('Number of Tickets')
                plt.tight_layout()
            elif "pie" in query:
                # Create a pie chart of ticket distribution by requestor seniority
                seniority_counts = self.df['Customer Age'].value_counts()
                ax.pie(seniority_counts, labels=seniority_counts.index, autopct='%1.1f%%')
                ax.set_title('Ticket Distribution by Requestor Seniority')
        
        elif "it owner" in query:
            if "bar" in query or "chart" in query:
                # Create a bar chart of tickets by IT owner
                owner_counts = self.df['Customer Name'].value_counts().head(10)  # Top 10 customers
                owner_counts.plot(kind='bar', ax=ax)
                ax.set_title('Top 10 IT Owners by Ticket Count')
                ax.set_xlabel('IT Owner ID')
                ax.set_ylabel('Number of Tickets')
                plt.tight_layout()
        
        elif "correlation" in query or "relationship" in query:
            # Create a correlation heatmap
            numeric_cols = ['Customer Age', 'Time to Resolution']
            correlation = self.df[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Between Numeric Variables')
            plt.tight_layout()
        
        else:
            # Default visualization - ticket counts by priority
            priority_counts = self.df['Ticket Priority'].value_counts()
            priority_counts.plot(kind='bar', ax=ax)
            ax.set_title('Tickets by Priority')
            ax.set_xlabel('Priority')
            ax.set_ylabel('Number of Tickets')
            plt.tight_layout()
        
        # Convert the figure to a base64 string
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Return HTML with the image
        return f'<img src="data:image/png;base64,{img_str}" alt="Visualization" style="max-width:100%;">'